import pandas as pd
import requests
import streamlit as st

# Backend API URL
API_URL = "http://127.0.0.1:8000"

# Define skill categories
skills = ["Java", "Net", "Python", "JavaScript"]

# Add custom CSS styling
def add_custom_css():
    st.markdown("""
        <style>
            /* Main Page Layout */
            .css-18e3th9 { background-color: #f5f5f5; }
            .css-1d391kg { color: #333333; }
            
            /* Sidebar Styling */
            .css-1lcbmhc, .stSidebar {  /* Targeting sidebar classes */
                background-color: white !important;  /* Change background to white */
                color: black !important;  /* Change text color to black for contrast */
            }
            .css-1lcbmhc h2 {
                color: black;  /* Change header color to black */
            }
            .css-1d391kg, .css-17eq0hr a {
                color: black;  /* Change link color to black */
            }

            /* Titles - Adjusted font sizes for a more professional look */
            h1 {
                color: #6a0dad;
                font-size: 30px; /* Smaller heading */
                font-weight: bold;
            }
            h2 {
                color: #6a0dad;
                font-size: 20px; /* Smaller subheading */
                font-weight: bold;
            }
            h3 {
                color: #4b0082;
                font-size: 20px;
            }

            /* Buttons */
            button {
                background-color: #4b0082; /* Indigo */
                color: white;
                border-radius: 8px;
            }
            button:hover {
                background-color: #9370db; /* Medium Purple */
                color: #ffffff;
            }

            /* Selectbox */
            .css-1n543e5 {
                background-color: #9370db;
                border-radius: 8px;
                color: white;
            }

            /* Upload file */
            .css-1x8cf1d {
                background-color: #6a5acd; /* Slate Blue */
                color: white;
                border-radius: 8px;
            }
            .css-1x8cf1d:hover {
                background-color: #836fff;
            }

            /* Icon Styling */
            .icon-style {
                font-size: 40px;
                color: #4b0082; /* Indigo */
            }
        </style>
    """, unsafe_allow_html=True)

# Apply the custom CSS
add_custom_css()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["SkillPath Data Ingestion & Viewing","SkillPath Vector Creation & Viewing", 
                                            "SkillPath FilterChat", "SkillPath Chat",
                                            "RoleBased Data Ingestion & Viewing",
                                            "RoleBased Vector Creation & Viewing", 
                                            "RoleBased FilterChat", "RoleBased Chat"])


if page == "SkillPath Data Ingestion & Viewing":
    st.title("Skill-based CSV Data Uploader & Viewer")

    # Upload Section
    st.header("Upload CSV Data")
    skill = st.selectbox("Select Skill Type", options=skills)
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])

    if st.button("Upload CSV"):
        if csv_file is not None and skill:
            files = {"file": (csv_file.name, csv_file.getvalue())}
            data = {"skill_type": skill}
            response = requests.post(f"{API_URL}/upload-csv/", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                inserted = result.get("inserted", 0)
                updated = result.get("updated", 0)
                no_change = result.get("no_change", 0)
                st.success(f"{skill} file upload successful! No Change: {no_change}, Inserted: {inserted} records, Updated: {updated} records.")
            else:
                st.error("Error uploading CSV. Please try again.")
        else:
            st.warning("Please select a skill type and upload a CSV file.")

    # Retrieve Section
    st.header("Retrieve Data by Skill")
    skill_to_view = st.selectbox("Select Skill Type to View Data", options=skills)
    print("skillview",skill_to_view)
    
    if st.button("Retrieve Data"):
        response = requests.get(f"{API_URL}/retrieve-data/", params={"skill_type": skill_to_view})
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.warning("No data found for the selected skill.")
        else:
            st.error("Error retrieving data.")

elif page == "Vector Creation & Search":
    st.title("Vector Creation & Search")

    # Create Vectors Section
    st.header("Create Vectors from Existing Data")
    skill_to_create_vectors = st.selectbox("Select Skill Type for Vector Creation", options=skills)

    if st.button("Create Vectors"):
        if skill_to_create_vectors:
            response = requests.post(f"{API_URL}/create-vectors-from-existing/", json={"skill_type": skill_to_create_vectors})
            if response.status_code == 200:
                st.success("Vectors created successfully.")
            else:
                st.error("Error creating vectors.")
        else:
            st.warning("Please select a skill type to create vectors.")

    # Search Vectors Section
    st.header("Similarity Search Vectors")
    vskill_ = st.selectbox("Select Skill ", options=skills)
    search_query = st.text_input("Enter your search query")
    path_type = st.selectbox("Filter by Path Type", ["All", "Upskilling", "Cross-Skilling"])
    current_proficiency = st.selectbox("Filter by Current Proficiency", ["Novice", "AdvancedBeginner", "Proficient", "Competent"])

    if st.button("Search"):
        if search_query:
            proficiency_filter = current_proficiency if current_proficiency != "All" else None
            path_type_filter = path_type if path_type != "All" else None
            
            response = requests.post(
                f"{API_URL}/search-vectors/",
                json={"skill_type": skill_to_create_vectors,"skill": vskill_, "query": search_query, "proficiency": path_type_filter, "path_type": path_type_filter}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    st.write("Search Results:")
                    for result in results:
                        st.write(f"**{result['page_content']}** ")
                else:
                    st.write("No results found.")
            else:
                st.error("Error searching vectors.")
        else:
            st.warning("Please enter a search query.")

elif page == "SkillPath Vector Creation & Viewing":
    st.title("Vector Creation & Retrieve")

    # Create Vectors Section
    st.header("Create Vectors from Existing Data")
    cskill_to_create_vectors = st.selectbox("Skill Type for Vector Creation", options=skills)

    if st.button("Create Vectors"):
        if cskill_to_create_vectors:
            response = requests.post(f"{API_URL}/create-vectors-crud-new/", json={"skill_type": cskill_to_create_vectors})
            if response.status_code == 200:
                st.success("Vectors created successfully.")
                result = response.json()
                inserted = result.get("inserted", 0)
                updated = result.get("updated", 0)
                no_change = result.get("no_change", 0)
                deleted = result.get("deleted", 0)
                st.success(f"{cskill_to_create_vectors} Vectors creation  successful! No Change: {no_change}, Inserted: {inserted} records, Updated: {updated} records, Deleted : {deleted}.")
            else:
                st.error("Error creating vectors.")
        else:
            st.warning("Please select a skill type to create vectors.")

           

    

    # Retrieve Section
    st.header("Retrieve Data by Skill")
    skill_to_view = st.selectbox("Skill Type to View Data", options=skills)
    
    print("skillview",skill_to_view)
    if st.button("Retrieve Data"):
        response = requests.get(f"{API_URL}/retrieve-data-vector/", params={"skill_type": skill_to_view})
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                # Display data as a table
                st.table(data)
            else:
                st.write("No data available for the selected skill.")
        else:
            st.error("Error retrieving data.")


elif page == "SkillPath FilterChat":
    st.title("Direct Search")

    
    # Search Vectors Section
    st.header("Direct Search Vectors")
    skill_ = st.selectbox("Select Skill ", options=skills)
    skill_type = st.selectbox("Select Skill Type ", options=skills)
    path_type = st.selectbox("Filter by Path Type", ["All", "Upskilling", "Cross-Skilling"])  # Update options as needed
    current_proficiency = st.selectbox("Filter by Current Proficiency", ["Novice",  "AdvancedBeginner","Proficient","Competent"])  # Update options as needed
    # search_query = st.text_input("Enter your search query")
    if st.button("Search"):
        if skill_ :
            proficiency_filter = current_proficiency if current_proficiency != "All" else None
            path_type_filter = path_type if path_type != "All" else None
            
            # Send POST request to search vectors
            response = requests.post(
                f"{API_URL}/direct_search-vectors/",
                json={"skill_type": skill_type,"skill": skill_, "proficiency" :proficiency_filter,"path_type" : path_type_filter}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    st.write("Search Results:")
                    for result in results:
                        st.write(f"**{result['metadata']}** ")
                else:
                    st.write("No results found.")
            else:
                st.error("Error searching vectors.")
        else:
            st.warning("Please enter a search query.")

elif page == "SkillPath Chat":
    st.title("Chat")

    # Chat Section
    st.header("Ask Question")

    # Create columns for horizontal layout
    col1,col2, col3, col4 = st.columns(4)

    # Place each select box in its own column
    with col1:
        cskill_type = st.selectbox("Select Skill Type", options=skills)
    with col2:
        cskill_ = st.selectbox("Select Skill", options=skills)
    with col3:
        
        current_proficiency = st.selectbox(" Proficiency", ["Novice", "AdvancedBeginner", "Proficient", "Competent"])
    with col4:
        path_type = st.selectbox(" Path Type", ["All", "Upskilling", "Cross-Skilling"])

    # Input for search query
    search_query = st.text_input("Enter your search query")

    # Button to submit the query
    if st.button("Search"):
        if search_query:
            proficiency_filter = current_proficiency if current_proficiency != "All" else None
            path_type_filter = path_type if path_type != "All" else None
            response = requests.post(
                f"{API_URL}/chat/",
                json={"skill_type": cskill_type,"skill": cskill_, "query": search_query, "proficiency": proficiency_filter, "path_type": path_type_filter}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    st.write("Search Results:")
                    st.write(f"** {results}** ")
                else:
                    st.write("No results found.")
            else:
                st.error("Error in chat response.")
        else:
            st.warning("Please enter a search query.")

elif page == "RoleBased Data Ingestion & Viewing":
    st.title("Role-based CSV Data Uploader & Viewer")

    # Upload Section
    st.header("Upload CSV Data")
    skill = st.selectbox("Select SkillType", options=skills)
    csv_file = st.file_uploader("Upload CSV File", type=["csv"])

    if st.button("Upload CSV"):
        if csv_file is not None and skill:
            files = {"file": (csv_file.name, csv_file.getvalue())}
            data = {"skill_type": skill}
            response = requests.post(f"{API_URL}/rolebased-upload-csv/", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                inserted = result.get("inserted", 0)
                updated = result.get("updated", 0)
                no_change = result.get("no_change", 0)
                st.success(f"{skill} file upload successful! No Change: {no_change}, Inserted: {inserted} records, Updated: {updated} records.")
            else:
                st.error("Error uploading CSV. Please try again.")
        else:
            st.warning("Please select a skill type and upload a CSV file.")

    # Retrieve Section
    st.header("Retrieve Data by Skill")
    skill_to_view = st.selectbox("Select Skill Type to View Data", options=skills)

    if st.button("Retrieve Data"):
        response = requests.get(f"{API_URL}/rolebased-retrieve-data/", params={"skill_type": skill_to_view})
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.warning("No data found for the selected skill.")
        else:
            st.error("Error retrieving data.")

elif page == "RoleBased Vector Creation & Viewing":
    st.title("Vector Creation & Retrieve")

    # Create Vectors Section
    st.header("Create Vectors from  Data")
    cskill_to_create_vectors = st.selectbox("SkillType for Vector Creation", options=skills)

    if st.button("Create Vectors"):
        if cskill_to_create_vectors:
            response = requests.post(f"{API_URL}/rolebased-create-vectors-crud-new/", json={"skill_type": cskill_to_create_vectors})
            if response.status_code == 200:
                st.success("Vectors created successfully.")
                result = response.json()
                inserted = result.get("inserted", 0)
                updated = result.get("updated", 0)
                no_change = result.get("no_change", 0)
                deleted = result.get("deleted", 0)
                st.success(f"{cskill_to_create_vectors} Vectors creation  successful! No Change: {no_change}, Inserted: {inserted} records, Updated: {updated} records, Deleted : {deleted}.")
            else:
                st.error("Error creating vectors.")
        else:
            st.warning("Please select a skill type to create vectors.")

           

    

    # Retrieve Section
    st.header("Retrieve Data by Skill")
    skill_to_view = st.selectbox("Skill Type to View Data", options=skills)
    
    print("skillview",skill_to_view)
    if st.button("Retrieve Data"):
        response = requests.get(f"{API_URL}/rolebased-retrieve-data-vector/", params={"skill_type": skill_to_view})
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                # Display data as a table
                st.table(data)
            else:
                st.write("No data available for the selected skill.")
        else:
            st.error("Error retrieving data.")


elif page == "RoleBased FilterChat":
    st.title("Transition Guidance Between Roles")

    # Chat Section
    st.header("Ask Question")

    # Create columns for horizontal layout
    col1, col2, col3 = st.columns([1, 1, 2])  # Adjust the column width ratios

    # Place each select box in its own column
    with col1:
        dcskill_type = st.selectbox("Select Skill Type", options=skills)
    with col2:
        dcurrent_role = st.selectbox("Current Role", ["Java Programmer", "Java Developer", "Java Senior Developer", "Java Team Lead", "Java Architect", "Java Senior Architect"])
    with col3:
        dcurrent_proficiency = st.selectbox("Current Proficiency Level", ["Novice", "Advanced Beginner", "Competent", "Proficient", "Expert"])

    # Create additional columns for target role and proficiency
    col4, col5 = st.columns([1, 1])  # Smaller width for the target role and proficiency
    with col4:
        dtarget_role = st.selectbox("Target Role", ["Java Programmer", "Java Developer", "Java Senior Developer", "Java Team Lead", "Java Architect", "Java Senior Architect"])
    with col5:
        dtarget_proficiency = st.selectbox("Target Proficiency Level", ["Novice", "Advanced Beginner", "Competent", "Proficient", "Expert"])

    # Input for search query
    #search_query = st.text_input("Enter your search query")

    # Button to submit the query
    if st.button("Guide"):
        if dcurrent_role:
            response = requests.post(
                f"{API_URL}/rolebased-Direct_chat/",
                json={
                    "skill_type": dcskill_type,
                    "current_role": dcurrent_role,
                    "proficiency": dcurrent_proficiency,
                    "target_role": dtarget_role,
                    "target_proficiency": dtarget_proficiency
                }
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    st.write("Search Results:")
                    st.write(f"**{results}** ")
                else:
                    st.write("No results found.")
            else:
                st.error("Error in chat response.")
        else:
            st.warning("Please enter a search query.")

elif page == "RoleBased Chat":
    st.title("Chat")

    # Chat Section
    st.header("Ask Question")

 
    cskill_type = st.selectbox("Select Skill Type", options=skills)
    
    # Input for search query
    search_query = st.text_input("Enter your query")

    # Button to submit the query
    if st.button("Search"):
        if search_query:
            response = requests.post(
                f"{API_URL}/rolebased-chat/",
                json={
                    "skill_type": cskill_type,
                    "query": search_query
                    
                }
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    st.write("Search Results:")
                    st.write(f"**{results}** ")
                else:
                    st.write("No results found.")
            else:
                st.error("Error in chat response.")
        else:
            st.warning("Please enter a search query.")