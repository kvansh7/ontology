import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from langchain.chat_models import init_chat_model
from process_vendor import process_vendor, match_vendors_to_solution, match_vendor_to_ps, process_solution
from dotenv import load_dotenv
# Page configuration
st.set_page_config(
    page_title="Vendor-Solution Matching System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.section-header {
    font-size: 1.5rem;
    color: #333;
    border-bottom: 2px solid #1f77b4;
    padding-bottom: 0.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.success-msg {
    background-color: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 0.375rem;
    border: 1px solid #c3e6cb;
    margin: 1rem 0;
}
.error-msg {
    background-color: #f8d7da;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 0.375rem;
    border: 1px solid #f5c6cb;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# MongoDB Configuration
load_dotenv()
def init_mongodb():
    """Initialize MongoDB connection"""
    try:
        # Update with your MongoDB connection string
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client["vendor_solution_db"]
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {str(e)}")
        return None

# Initialize MongoDB
db = init_mongodb()

# Collections
if db is not None:
    problems_collection = db.problems
    vendors_collection = db.vendors
    solutions_collection = db.solutions
    processed_solutions_collection = db.processed_solutions
    fs = gridfs.GridFS(db)

# Initialize language model

def init_language_model():
    """Initialize language model"""
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        return model
    except Exception as e:
        st.error(f"Failed to initialize language model: {str(e)}")
        return None

model = init_language_model()

# Utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {"pdf", "docx", "pptx"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file):
    """Save uploaded file to GridFS and return file_id"""
    try:
        file_id = fs.put(
            uploaded_file.getvalue(),
            filename=uploaded_file.name,
            content_type=uploaded_file.type
        )
        return str(file_id)
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def get_file_from_gridfs(file_id):
    """Retrieve file from GridFS"""
    try:
        file_data = fs.get(ObjectId(file_id))
        return file_data
    except Exception as e:
        st.error(f"Error retrieving file: {str(e)}")
        return None

# Core functions
def submit_problem(problem_data, uploaded_files):
    """Submit a problem to MongoDB"""
    try:
        # Save uploaded files to GridFS
        saved_files = []
        for file in uploaded_files:
            if file and allowed_file(file.name):
                file_id = save_uploaded_file(file)
                if file_id:
                    saved_files.append({
                        "filename": file.name,
                        "file_id": file_id,
                        "content_type": file.type
                    })

        # Build problem document
        problem_doc = {
            "title": problem_data.get("title", ""),
            "description": problem_data.get("description", ""),
            "priority": problem_data.get("priority", ""),
            "expectedOutcome": problem_data.get("expectedOutcome", ""),
            "contactEmail": problem_data.get("contactEmail", ""),
            "timeline": problem_data.get("timeline", ""),
            "website": problem_data.get("website", ""),
            "documents": saved_files,
            "created_at": datetime.now(),
            "status": "submitted"
        }

        # Insert problem
        problem_result = problems_collection.insert_one(problem_doc)
        problem_id = problem_result.inserted_id

        # Generate solution using Gemini
        if model:
            prompt = f"""
This is a problem statement provided:

Title: {problem_data.get('title', '')}
Description: {problem_data.get('description', '')}
Priority: {problem_data.get('priority', '')}
Expected Outcome: {problem_data.get('expectedOutcome', '')}
Timeline: {problem_data.get('timeline', '')}
Website: {problem_data.get('website', '')}

Task:
Give me a detailed solution on how to build this, 
including all technical details, necessary tools, frameworks, 
libraries, and technology keywords. Write the answer in a 
clear technical paragraph.
"""
            response = model.invoke(prompt)
            solution_text = (
                response.content
                if hasattr(response, "content") and response.content
                else str(response)
            )

            # Save solution
            solution_doc = {
                "problem_id": problem_id,
                "title": problem_data["title"],
                "solution": solution_text,
                "created_at": datetime.now()
            }
            solution_result = solutions_collection.insert_one(solution_doc)
            
            return {
                "status": "success",
                "problem_id": str(problem_id),
                "solution_id": str(solution_result.inserted_id),
                "problem": problem_doc,
                "solution": solution_doc
            }
        else:
            return {
                "status": "success",
                "problem_id": str(problem_id),
                "problem": problem_doc,
                "message": "Problem saved but solution generation failed"
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def submit_vendor(vendor_data, uploaded_files):
    """Submit a vendor to MongoDB"""
    try:
        # Save uploaded files to temporary location for processing
        file_paths = []
        for file in uploaded_files:
            if file and allowed_file(file.name):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.getvalue())
                    file_paths.append(tmp_file.name)

        # Process the first document
        processed_data = process_vendor(file_paths[0]) if file_paths else {
            "keywords": [],
            "domain_profile": {},
            "subdomains": [],
            "embedding": np.zeros(384).tolist()
        }

        # Save files to GridFS
        saved_files = []
        for file in uploaded_files:
            if file and allowed_file(file.name):
                file_id = save_uploaded_file(file)
                if file_id:
                    saved_files.append({
                        "filename": file.name,
                        "file_id": file_id,
                        "content_type": file.type
                    })

        # Build vendor document
        vendor_doc = {
            "vendorName": vendor_data.get("vendorName", ""),
            "contactEmail": vendor_data.get("contactEmail", ""),
            "contactPhone": vendor_data.get("contactPhone", ""),
            "website": vendor_data.get("website", ""),
            "domainExpertise": vendor_data.get("domainExpertise", ""),
            "internalRating": vendor_data.get("internalRating", ""),
            "comments": vendor_data.get("comments", ""),
            "keywords": processed_data["keywords"],
            "domain_profile": processed_data["domain_profile"],
            "subdomains": processed_data["subdomains"],
            "embedding": processed_data["embedding"].tolist() if isinstance(processed_data["embedding"], np.ndarray) else processed_data["embedding"],
            "documents": saved_files,
            "created_at": datetime.now()
        }

        # Insert vendor
        result = vendors_collection.insert_one(vendor_doc)

        # Clean up temporary files
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except:
                pass

        return {
            "status": "success",
            "vendor_id": str(result.inserted_id),
            "message": "Vendor profile saved successfully"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_all_problems():
    """Get all problems from MongoDB"""
    try:
        problems = list(problems_collection.find())
        for problem in problems:
            problem["_id"] = str(problem["_id"])
        return problems
    except Exception as e:
        st.error(f"Error fetching problems: {str(e)}")
        return []

def get_all_vendors():
    """Get all vendors from MongoDB"""
    try:
        vendors = list(vendors_collection.find())
        for vendor in vendors:
            vendor["_id"] = str(vendor["_id"])
        return vendors
    except Exception as e:
        st.error(f"Error fetching vendors: {str(e)}")
        return []

def get_all_solutions():
    """Get all solutions from MongoDB"""
    try:
        solutions = list(solutions_collection.find())
        for solution in solutions:
            solution["_id"] = str(solution["_id"])
        return solutions
    except Exception as e:
        st.error(f"Error fetching solutions: {str(e)}")
        return []

def match_solution_to_vendors(solution_id):
    """Match a solution to vendors"""
    try:
        # Get solution
        solution = solutions_collection.find_one({"_id": ObjectId(solution_id)})
        if not solution:
            return {"error": "Solution not found"}

        # Get all vendors
        vendors = list(vendors_collection.find())
        if not vendors:
            return {"error": "No vendors available"}

        # Match vendors
        results = []
        for vendor in vendors:
            # Create temporary files for matching function
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as vendor_file:
                json.dump(vendor, vendor_file, default=str)
                vendor_file_path = vendor_file.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as solution_file:
                json.dump(solution, solution_file, default=str)
                solution_file_path = solution_file.name

            try:
                match_result = match_vendors_to_solution([vendor_file_path], solution_file_path)
                results.extend(match_result)
            finally:
                # Clean up temporary files
                os.unlink(vendor_file_path)
                os.unlink(solution_file_path)

        return {"solution_id": solution_id, "matches": results}

    except Exception as e:
        return {"error": str(e)}

def process_solution_and_match(solution_data):
    """Process solution and match with vendors"""
    try:
        # Process solution
        processed_solution = process_solution(solution_data)

        # Get all vendors
        vendors = list(vendors_collection.find())
        if not vendors:
            return {"error": "No vendors available"}

        # Match vendors to processed solution
        vendor_matches = []
        for i, vendor_data in enumerate(vendors):
            match_result = match_vendor_to_ps(vendor_data, processed_solution)

            vendor_id = f"VENDOR_{i+1:03d}"
            vendor_name = vendor_data.get("vendorName", "Unknown")

            vendor_matches.append({
                'vendor_id': vendor_id,
                'vendor_name': vendor_name,
                'final_score': match_result['final_score'],
                'component_scores': match_result['component_scores'],
                'matched_domains': match_result['matched_subdomains'],
                'matched_tools': match_result['matched_tools'],
                'justification': match_result['justification']
            })

        # Sort matches by final_score descending
        vendor_matches.sort(key=lambda x: x['final_score'], reverse=True)

        # Add vendor matches to processed solution
        processed_solution["vendor_matches"] = vendor_matches
        processed_solution["created_at"] = datetime.now()

        # Save processed solution
        result = processed_solutions_collection.insert_one(processed_solution)

        return {
            "status": "success",
            "processed_solution_id": str(result.inserted_id),
            "vendor_matches": vendor_matches
        }

    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
def main():
    st.markdown('<h1 class="main-header">🔧 Vendor-Solution Matching System</h1>', unsafe_allow_html=True)
    
    if db is None:
        st.error("❌ Database connection failed. Please check your MongoDB configuration.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Submit Problem", "Submit Vendor", "View Problems", "View Vendors", "View Solutions", "Process & Match"]
    )

    if page == "Dashboard":
        show_dashboard()
    elif page == "Submit Problem":
        show_submit_problem()
    elif page == "Submit Vendor":
        show_submit_vendor()
    elif page == "View Problems":
        show_view_problems()
    elif page == "View Vendors":
        show_view_vendors()
    elif page == "View Solutions":
        show_view_solutions()
    elif page == "Process & Match":
        show_process_and_match()

def show_dashboard():
    """Dashboard page"""
    st.markdown('<h2 class="section-header">📊 Dashboard</h2>', unsafe_allow_html=True)
    
    # Get statistics
    problems_count = problems_collection.count_documents({})
    vendors_count = vendors_collection.count_documents({})
    solutions_count = solutions_collection.count_documents({})
    processed_solutions_count = processed_solutions_collection.count_documents({})
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Problems", problems_count)
    
    with col2:
        st.metric("Total Vendors", vendors_count)
    
    with col3:
        st.metric("Total Solutions", solutions_count)
    
    with col4:
        st.metric("Processed Solutions", processed_solutions_count)
    
    # Recent activity
    st.markdown('<h3 class="section-header">📈 Recent Activity</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Problems")
        recent_problems = list(problems_collection.find().sort("created_at", -1).limit(5))
        if recent_problems:
            for problem in recent_problems:
                st.write(f"• {problem.get('title', 'Untitled')} - {problem.get('priority', 'No priority')}")
        else:
            st.write("No problems submitted yet")
    
    with col2:
        st.subheader("Recent Vendors")
        recent_vendors = list(vendors_collection.find().sort("created_at", -1).limit(5))
        if recent_vendors:
            for vendor in recent_vendors:
                st.write(f"• {vendor.get('vendorName', 'Unknown')} - {vendor.get('domainExpertise', 'No domain specified')}")
        else:
            st.write("No vendors registered yet")

def show_submit_problem():
    """Submit problem page"""
    st.markdown('<h2 class="section-header">📝 Submit Problem</h2>', unsafe_allow_html=True)
    
    with st.form("problem_form"):
        st.subheader("Problem Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Problem Title *", placeholder="Enter problem title")
            priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            contact_email = st.text_input("Contact Email *", placeholder="your.email@company.com")
            timeline = st.text_input("Timeline", placeholder="e.g., 3 months")
        
        with col2:
            expected_outcome = st.text_area("Expected Outcome", placeholder="Describe expected results")
            website = st.text_input("Website", placeholder="https://yourcompany.com")
        
        description = st.text_area("Problem Description *", placeholder="Detailed description of the problem", height=150)
        
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload supporting documents",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True
        )
        
        submitted = st.form_submit_button("Submit Problem", type="primary")
        
        if submitted:
            if not title or not description or not contact_email:
                st.error("Please fill in all required fields marked with *")
            else:
                problem_data = {
                    "title": title,
                    "description": description,
                    "priority": priority,
                    "expectedOutcome": expected_outcome,
                    "contactEmail": contact_email,
                    "timeline": timeline,
                    "website": website
                }
                
                with st.spinner("Submitting problem and generating solution..."):
                    result = submit_problem(problem_data, uploaded_files or [])
                
                if result["status"] == "success":
                    st.markdown('<div class="success-msg">✅ Problem submitted successfully!</div>', unsafe_allow_html=True)
                    st.write(f"**Problem ID:** {result['problem_id']}")
                    if "solution_id" in result:
                        st.write(f"**Solution ID:** {result['solution_id']}")
                        st.subheader("Generated Solution")
                        st.write(result["solution"]["solution"])
                else:
                    st.markdown(f'<div class="error-msg">❌ Error: {result["message"]}</div>', unsafe_allow_html=True)

def show_submit_vendor():
    """Submit vendor page"""
    st.markdown('<h2 class="section-header">🏢 Submit Vendor</h2>', unsafe_allow_html=True)
    
    with st.form("vendor_form"):
        st.subheader("Vendor Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vendor_name = st.text_input("Vendor Name *", placeholder="Company Name")
            contact_email = st.text_input("Contact Email *", placeholder="contact@company.com")
            contact_phone = st.text_input("Contact Phone", placeholder="+1-xxx-xxx-xxxx")
            website = st.text_input("Website", placeholder="https://company.com")
        
        with col2:
            domain_expertise = st.text_area("Domain Expertise", placeholder="Areas of expertise")
            internal_rating = st.selectbox("Internal Rating", ["1", "2", "3", "4", "5"])
            comments = st.text_area("Comments", placeholder="Additional notes")
        
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload vendor documents (company profile, portfolio, etc.)",
            type=["pdf", "docx", "pptx"],
            accept_multiple_files=True
        )
        
        submitted = st.form_submit_button("Submit Vendor", type="primary")
        
        if submitted:
            if not vendor_name or not contact_email:
                st.error("Please fill in all required fields marked with *")
            else:
                vendor_data = {
                    "vendorName": vendor_name,
                    "contactEmail": contact_email,
                    "contactPhone": contact_phone,
                    "website": website,
                    "domainExpertise": domain_expertise,
                    "internalRating": internal_rating,
                    "comments": comments
                }
                
                with st.spinner("Processing vendor profile..."):
                    result = submit_vendor(vendor_data, uploaded_files or [])
                
                if result["status"] == "success":
                    st.markdown('<div class="success-msg">✅ Vendor profile saved successfully!</div>', unsafe_allow_html=True)
                    st.write(f"**Vendor ID:** {result['vendor_id']}")
                else:
                    st.markdown(f'<div class="error-msg">❌ Error: {result["message"]}</div>', unsafe_allow_html=True)

def show_view_problems():
    """View problems page"""
    st.markdown('<h2 class="section-header">📋 View Problems</h2>', unsafe_allow_html=True)
    
    problems = get_all_problems()
    
    if problems:
        st.write(f"**Total Problems:** {len(problems)}")
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("🔍 Search problems", placeholder="Search by title or description")
        with col2:
            priority_filter = st.selectbox("Filter by Priority", ["All", "Low", "Medium", "High", "Critical"])
        
        # Filter problems
        filtered_problems = problems
        if search_term:
            filtered_problems = [p for p in filtered_problems if 
                               search_term.lower() in p.get('title', '').lower() or 
                               search_term.lower() in p.get('description', '').lower()]
        
        if priority_filter != "All":
            filtered_problems = [p for p in filtered_problems if p.get('priority') == priority_filter]
        
        # Display problems
        for problem in filtered_problems:
            with st.expander(f"📋 {problem.get('title', 'Untitled')} - {problem.get('priority', 'No priority')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {problem.get('description', 'No description')}")
                    st.write(f"**Expected Outcome:** {problem.get('expectedOutcome', 'Not specified')}")
                    st.write(f"**Timeline:** {problem.get('timeline', 'Not specified')}")
                
                with col2:
                    st.write(f"**Contact Email:** {problem.get('contactEmail', 'Not provided')}")
                    st.write(f"**Website:** {problem.get('website', 'Not provided')}")
                    st.write(f"**Created:** {problem.get('created_at', 'Unknown')}")
                
                if problem.get('documents'):
                    st.write(f"**Documents:** {len(problem['documents'])} file(s) uploaded")
    else:
        st.info("No problems found. Submit a problem to get started!")

def show_view_vendors():
    """View vendors page"""
    st.markdown('<h2 class="section-header">🏢 View Vendors</h2>', unsafe_allow_html=True)
    
    vendors = get_all_vendors()
    
    if vendors:
        st.write(f"**Total Vendors:** {len(vendors)}")
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("🔍 Search vendors", placeholder="Search by name or expertise")
        with col2:
            rating_filter = st.selectbox("Filter by Rating", ["All", "1", "2", "3", "4", "5"])
        
        # Filter vendors
        filtered_vendors = vendors
        if search_term:
            filtered_vendors = [v for v in filtered_vendors if 
                              search_term.lower() in v.get('vendorName', '').lower() or 
                              search_term.lower() in v.get('domainExpertise', '').lower()]
        
        if rating_filter != "All":
            filtered_vendors = [v for v in filtered_vendors if v.get('internalRating') == rating_filter]
        
        # Display vendors
        for vendor in filtered_vendors:
            with st.expander(f"🏢 {vendor.get('vendorName', 'Unknown')} - Rating: {vendor.get('internalRating', 'N/A')}⭐"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Domain Expertise:** {vendor.get('domainExpertise', 'Not specified')}")
                    st.write(f"**Contact Email:** {vendor.get('contactEmail', 'Not provided')}")
                    st.write(f"**Contact Phone:** {vendor.get('contactPhone', 'Not provided')}")
                
                with col2:
                    st.write(f"**Website:** {vendor.get('website', 'Not provided')}")
                    st.write(f"**Comments:** {vendor.get('comments', 'No comments')}")
                    st.write(f"**Created:** {vendor.get('created_at', 'Unknown')}")
                
                # Show processed data
                if vendor.get('keywords'):
                    st.write(f"**Keywords:** {', '.join(vendor['keywords'][:10])}...")
                
                if vendor.get('subdomains'):
                    st.write(f"**Subdomains:** {', '.join(vendor['subdomains'][:5])}...")
    else:
        st.info("No vendors found. Submit a vendor to get started!")

def show_view_solutions():
    """View solutions page"""
    st.markdown('<h2 class="section-header">💡 View Solutions</h2>', unsafe_allow_html=True)
    
    solutions = get_all_solutions()
    
    if solutions:
        st.write(f"**Total Solutions:** {len(solutions)}")
        
        search_term = st.text_input("🔍 Search solutions", placeholder="Search by title")
        
        # Filter solutions
        filtered_solutions = solutions
        if search_term:
            filtered_solutions = [s for s in filtered_solutions if 
                                search_term.lower() in s.get('title', '').lower()]
        
        # Display solutions
        for solution in filtered_solutions:
            with st.expander(f"💡 {solution.get('title', 'Untitled Solution')}"):
                st.write(f"**Created:** {solution.get('created_at', 'Unknown')}")
                st.markdown("**Solution:**")
                st.write(solution.get('solution', 'No solution text'))
                
                # Show match button
                if st.button(f"🔍 Find Matching Vendors", key=f"match_{solution['_id']}"):
                    with st.spinner("Finding matching vendors..."):
                        match_result = match_solution_to_vendors(solution['_id'])
                    
                    if "error" not in match_result:
                        st.subheader("Matching Vendors")
                        if match_result.get('matches'):
                            for match in match_result['matches']:
                                st.write(f"**Vendor:** {match.get('vendor_name', 'Unknown')}")
                                st.write(f"**Score:** {match.get('final_score', 0):.2f}")
                                st.write("---")
                        else:
                            st.info("No matching vendors found")
                    else:
                        st.error(f"Error: {match_result['error']}")
    else:
        st.info("No solutions found. Submit a problem to generate solutions!")

def show_process_and_match():
    """Process and match page"""
    st.markdown('<h2 class="section-header">⚙️ Process Solution & Match Vendors</h2>', unsafe_allow_html=True)
    
    st.info("This feature processes a solution with advanced analysis and finds the best matching vendors.")
    
    solutions = get_all_solutions()
    
    if solutions:
        # Solution selector
        solution_options = {s['_id']: s.get('title', 'Untitled') for s in solutions}
        selected_solution_id = st.selectbox(
            "Select a solution to process and match",
            options=list(solution_options.keys()),
            format_func=lambda x: solution_options[x]
        )
        
        # Show selected solution
        selected_solution = next(s for s in solutions if s['_id'] == selected_solution_id)
        with st.expander("📄 Selected Solution Details"):
            st.write(f"**Title:** {selected_solution.get('title', 'Untitled')}")
            st.write(f"**Solution Text:** {selected_solution.get('solution', 'No solution text')[:500]}...")
        
        if st.button("⚙️ Process & Match", type="primary"):
            with st.spinner("Processing solution and matching vendors..."):
                # Prepare solution data
                solution_data = {
                    "title": selected_solution.get('title', ''),
                    "solution": selected_solution.get('solution', '')
                }
                
                result = process_solution_and_match(solution_data)
            
            if "error" not in result:
                st.success("✅ Processing and matching completed!")
                
                st.subheader(f"📊 Vendor Matches for: {solution_data['title']}")
                
                if result.get('vendor_matches'):
                    # Create summary DataFrame
                    matches_data = []
                    for match in result['vendor_matches']:
                        matches_data.append({
                            'Vendor ID': match.get('vendor_id', 'Unknown'),
                            'Vendor Name': match.get('vendor_name', 'Unknown'),
                            'Final Score': round(match.get('final_score', 0), 3),
                            'Domain Score': round(match.get('component_scores', {}).get('domain_score', 0), 3),
                            'Tool Score': round(match.get('component_scores', {}).get('tool_score', 0), 3),
                            'Matched Domains': len(match.get('matched_domains', [])),
                            'Matched Tools': len(match.get('matched_tools', []))
                        })
                    
                    df = pd.DataFrame(matches_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Show top matches in detail
                    st.subheader("🏆 Top 5 Matches")
                    top_matches = result['vendor_matches'][:5]
                    
                    for i, match in enumerate(top_matches, 1):
                        with st.expander(f"#{i} {match.get('vendor_name', 'Unknown')} - Score: {match.get('final_score', 0):.3f}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Final Score", f"{match.get('final_score', 0):.3f}")
                                component_scores = match.get('component_scores', {})
                                st.write("**Component Scores:**")
                                for key, value in component_scores.items():
                                    st.write(f"• {key.replace('_', '').title()}: {value:.3f}")
                            
                            with col2:
                                matched_domains = match.get('matched_domains', [])
                                matched_tools = match.get('matched_tools', [])
                                
                                if matched_domains:
                                    st.write(f"**Matched Domains ({len(matched_domains)}):**")
                                    for domain in matched_domains[:5]:
                                        st.write(f"• {domain}")
                                    if len(matched_domains) > 5:
                                        st.write(f"... and {len(matched_domains) - 5} more")
                                
                                if matched_tools:
                                    st.write(f"**Matched Tools ({len(matched_tools)}):**")
                                    for tool in matched_tools[:5]:
                                        st.write(f"• {tool}")
                                    if len(matched_tools) > 5:
                                        st.write(f"... and {len(matched_tools) - 5} more")
                            
                            # Justification
                            if match.get('justification'):
                                st.write("**AI Justification:**")
                                st.write(match['justification'])
                    
                    # Download results
                    st.subheader("💾 Export Results")
                    json_str = json.dumps(result, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Results as JSON",
                        data=json_str,
                        file_name=f"vendor_matches_{solution_data['title'].replace(' ', '_')}.json",
                        mime="application/json"
                    )
                    
                    # Save to processed solutions
                    st.info(f"✅ Results saved as processed solution with ID: {result.get('processed_solution_id')}")
                    
                else:
                    st.warning("No vendor matches found")
            else:
                st.error(f"❌ Error: {result['error']}")
    else:
        st.info("No solutions available for processing. Submit problems to generate solutions first!")

    # Show processed solutions history
    st.markdown('<h3 class="section-header">📚 Processed Solutions History</h3>', unsafe_allow_html=True)
    
    try:
        processed_solutions = list(processed_solutions_collection.find().sort("created_at", -1))
        
        if processed_solutions:
            for ps in processed_solutions:
                with st.expander(f"📊 {ps.get('title', 'Untitled')} - {len(ps.get('vendor_matches', []))} matches"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Created:** {ps.get('created_at', 'Unknown')}")
                        st.write(f"**Total Matches:** {len(ps.get('vendor_matches', []))}")
                    
                    with col2:
                        if ps.get('vendor_matches'):
                            best_match = ps['vendor_matches'][0]
                            st.write(f"**Best Match:** {best_match.get('vendor_name', 'Unknown')}")
                            st.write(f"**Best Score:** {best_match.get('final_score', 0):.3f}")
                    
                    # Download option
                    json_str = json.dumps(ps, indent=2, default=str)
                    st.download_button(
                        label="📥 Download",
                        data=json_str,
                        file_name=f"processed_{ps.get('title', 'solution').replace(' ', '_')}.json",
                        mime="application/json",
                        key=f"download_{ps['_id']}"
                    )
        else:
            st.info("No processed solutions found")
            
    except Exception as e:
        st.error(f"Error loading processed solutions: {str(e)}")

if __name__ == "__main__":
    main()
