# backend.py
#uvicorn backend:app --reload
#uvicorn backend:app --reload --log-level debug
#pip install streamlit fastapi pymongo motor pandas
#pip install langchain pymongo

import os
from dotenv import load_dotenv


load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pymongo import MongoClient
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List
import csv
import io
import numpy as np
from datetime import datetime
from pymongo import UpdateOne ,DeleteOne
from pymongo.collection import Collection
from fastapi.responses import JSONResponse
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

from langchain_core.documents import Document
#from bson import ObjectId
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
#from bson import ObjectId  # Import ObjectId from bson
import urllib 
import math

app = FastAPI()

# # MongoDB connection
# client = MongoClient(os.environ["MONGO_URI"])
# db = client["skill_data"]

model = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        verbose=False,
        temperature=0.3,
    )
COSMOS_MONGO_USER =os.getenv("COSMOS_MONGO_USER")
COSMOS_MONGO_PWD =os.getenv("COSMOS_MONGO_PWD")
COSMOS_MONGO_SERVER =os.getenv("COSMOS_MONGO_SERVER")
CMONGO_URI = "mongodb+srv://"+urllib.parse.quote(COSMOS_MONGO_USER)+":"+urllib.parse.quote(COSMOS_MONGO_PWD)+"@"+COSMOS_MONGO_SERVER+"?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
#client = MongoClient(mongo_conn)
# MongoDB connection
#client = AsyncIOMotorClient(os.environ["CMONGO_URI"])
client = AsyncIOMotorClient(CMONGO_URI)
db = client.skill_data

# Initialize  embeddings
embeddings = AzureOpenAIEmbeddings(azure_deployment="dna-text-embedding-ada-002")


# Define skill categories
skills = ["Java", "Net", "Python", "JavaScript"]


@app.post("/upload-csv/")
async def upload_csv(skill_type: str = Form(...), file: UploadFile = File(...)):
   

    # Read the CSV file into a DataFrame
    df = pd.read_csv(io.StringIO((await file.read()).decode("utf-8")))
    

    # Define the collection based on the skill type
    collection = db[skill_type]

    # Define the current date for insertion and update timestamps
    current_date = datetime.utcnow()

    # Track inserted, updated, and no change counts
    inserted_count = 0
    updated_count = 0
    no_change_count = 0

    # Prepare bulk operations
    bulk_operations = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Define filter criteria based on relevant fields
        filter_criteria = {
            "Path Name": row["Path Name"],
            "Current Skill": row["Current Skill"],
            "Current Proficiency": row["Current Proficiency"]
        }

        # Retrieve the existing document for comparison
        existing_doc = await collection.find_one(filter_criteria)

        if existing_doc is not None:
            # Compare fields to check for changes
            update_fields = {}
            for field in ["Path Type", "Recommended", "Prerequisites", "Learning Resources", 
                          "What you will learn", "Estimated Duration or Course Duration"]:
                if existing_doc.get(field) != row.get(field):
                    update_fields[field] = row.get(field)

            if update_fields:  # If there are fields to update
                update_fields["last_updated"] = current_date  # Add last_updated
                update_fields["status"] = "update"  # Set status to update
                bulk_operations.append(UpdateOne(
                    filter_criteria,
                    {
                        "$set": update_fields
                    }
                ))
                updated_count += 1  # Increment updated count
            else:
                # If there are no changes, update status to nochange
                bulk_operations.append(UpdateOne(
                    filter_criteria,
                    {
                        "$set": {"status": "nochange"}
                    }
                ))
                no_change_count += 1  # Increment no change count
        else:
            # Document does not exist; prepare an insert operation
            insert_data = {
                "Path Name": row["Path Name"],
                "Current Skill": row["Current Skill"],
                "Current Proficiency": row["Current Proficiency"],
                "Path Type": row.get("Path Type"),
                "Recommended": row.get("Recommended"),
                "Prerequisites": row.get("Prerequisites"),
                "Learning Resources": row.get("Learning Resources"),
                "What you will learn": row.get("What you will learn"),
                "Estimated Duration or Course Duration": row.get("Estimated Duration or Course Duration"),
                "last_updated": current_date,
                "status": "insert"  # Set status to insert
            }
            bulk_operations.append(UpdateOne(
                filter_criteria,
                {
                    "$set": insert_data,
                    "$setOnInsert": {"date_added": current_date}
                },
                upsert=True  # Ensure the document gets inserted if it doesn't exist
            ))
            #inserted_count += 1  # Increment inserted count

    # Execute bulk write if there are any operations to perform
    if bulk_operations:
        result = await collection.bulk_write(bulk_operations)
        # Increment inserted count for the new documents that were inserted
        inserted_count += len(result.upserted_ids)

    # Return the counts of the operations performed
    return {
        "status": "success",
        "inserted": inserted_count,
        "updated": updated_count,
        "no_change": no_change_count
    }
# Helper function to convert ObjectId to string and handle NaN/Infinity values
def clean_data_for_json_old(data):
    cleaned_data = []
    for item in data:
        # Convert ObjectId to string
        item["_id"] = str(item["_id"])

        # Replace non-JSON-compliant values
        for key, value in item.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                item[key] = None  # Replace NaN or Inf with None
        cleaned_data.append(item)
    
    return cleaned_data

def clean_data_for_json(data):
    cleaned_data = []
    for item in data:
        # Convert ObjectId to string for serialization
        if "_id" in item:
            item["_id"] = str(item["_id"])

        # Replace non-JSON-compliant float values
        for key, value in item.items():
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):  # Check for NaN or Inf
                    item[key] = None
            elif isinstance(value, dict):  # Recursively clean nested dictionaries
                item[key] = clean_data_for_json([value])[0]
            elif isinstance(value, list):  # Recursively clean lists
                item[key] = clean_data_for_json(value)
        cleaned_data.append(item)
    
    return cleaned_data

@app.get("/retrieve-data/")
async def retrieve_data(skill_type: str):
    print("SkillTYpe",skill_type)
    # if skill_type not in skills:
    #     return {"error": "Invalid skill type"}
    
    collection = db[skill_type]
    data = await collection.find().to_list(length=100)  # Limit the results
    print("BEFORE CLEAN DATA",data)
    
    # Clean data for JSON serialization
    data = clean_data_for_json(data)
    
    return {"data": data}

@app.get("/retrieve-data-vector/")
async def retrieve_data_vector(skill_type: str):
    print("SkillTYpe",skill_type)
    # client = MongoClient(os.environ["MONGO_URI"])
    #client = AsyncIOMotorClient(os.environ["CMONGO_URI"])
    client = AsyncIOMotorClient(CMONGO_URI)
    # if skill_type not in skills:
    #     return {"error": "Invalid skill type"}
    db = client.skill_data
    collection =  db[f"{skill_type}_collection"]
    data = await collection.find().to_list(length=100)  # Limit the results
    
    # Clean data for JSON serialization
    data = clean_data_for_json(data[0])
    print("DATA:",data)
    
    return {"data": data}

# Request model for vector creation
class VectorCreationRequest(BaseModel):
    skill_type: str
    

@app.post("/create-vectors-from-existing/")
async def create_vectors(request: VectorCreationRequest):
    #client = MongoClient(os.environ["CMONGO_URI"])
    client = MongoClient(CMONGO_URI)
    db = client.skill_data
    skill=request.skill_type
    collection = db[skill]
    data = []
    VECTOR_SEARCH_INDEX_NAME = "vector_index"
    
    # Use async for loop to iterate over the cursor
    for record in collection.find():
        data.append(record)

    if data:
        documents = []
        for record in data:
            # Create embedding document excluding 'status' and 'date'
            page_content = (
                f"if your current skill is {record['Current Skill']} and proficiency level is  {record['Current Proficiency']} "
                f" you can {record['Path Type']} yourself with Path {record['Path Name']} and recommended details {record['Recommended']} . The prerequisites is {record['Prerequisites']} and your learning resources are : '{record['Learning Resources']}' "
                f"What you will learn in this courses '{record['What you will learn']}' and estimated time to these courses are {record['Estimated Duration or Course Duration']}"
            )
            doc = Document(page_content=page_content, metadata={
                "Path Name": record["Path Name"],
                "Current Skill": record['Current Skill'],
                "Path Type": record['Path Type'],
                "Current Proficiency": record['Current Proficiency'],
                "Recommended": record['Recommended'],
                "Prerequisites": record['Prerequisites'],
                "Learning Resources": record['Learning Resources'],
                "What you will learn": record['What you will learn'],
                "Estimated Duration or Course Duration": record['Estimated Duration or Course Duration']
            })
            documents.append(doc)
        print("documents:",documents[0])
        #documents
        collection = db[f"{skill}_collection"]
        # Create embeddings and store them in a new collection
        #vectorstore = MongoDBAtlasVectorSearch(embeddings, collection_name=f"{skill}_collection")
        # Ensure MongoDBAtlasVectorSearch is properly initialized with MongoDB connection
        # Ensure the MongoDBAtlasVectorSearch uses async handling
    #     _ = MongoDBAtlasVectorSearch.from_documents(
    #     documents=documents,
    #     embedding=embeddings,
    #     collection=collection,
    #     index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    # )

        vector_store: AzureCosmosDBVectorSearch = AzureCosmosDBVectorSearch.from_documents(
                documents=documents,
                embedding=embeddings,
                collection=collection,
                index_name=VECTOR_SEARCH_INDEX_NAME,
            )

        #vectorstore.add_documents(documents)

        return JSONResponse(content={"message": "Vectors created successfully."})
    return JSONResponse(content={"message": "No data found."}, status_code=404)
def generate_embedding_content(record):
    return (
        f"If your current skill is {record['Current Skill']} and proficiency level is {record['Current Proficiency']}, "
        f"you can {record['Path Type']} yourself with Path {record['Path Name']} and recommended details {record['Recommended']}. "
        f"The prerequisites are {record['Prerequisites']} and your learning resources are: '{record['Learning Resources']}'. "
        f"What you will learn in this course is '{record['What you will learn']}' and the estimated time is {record['Estimated Duration or Course Duration']}."
    )


@app.post("/create-vectors-crud-new/")
async def create_vectors(request: VectorCreationRequest):
    #client = MongoClient(os.environ["CMONGO_URI"])
    client = MongoClient(CMONGO_URI)
    #client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = client.skill_data
    skill=request.skill_type
    #collection = db[skill]
    data = []
    #embedding_collection = db[f"{skill}_collection"]
    VECTOR_SEARCH_INDEX_NAME = "vector_index"
    # Define the current date for insertion and update timestamps
    current_date = datetime.utcnow()

    # Track inserted, updated, and no change counts
    inserted_count = 0
    updated_count = 0
    deleted_count = 0
    no_change_count = 0

    db = client.skill_data
   
    embedding_collection_name = f"{skill}_collection"
    print("COLLECTION NAME",embedding_collection_name)
    collection = db[skill]
    embedding_collection = db[f"{skill}_collection"]
    print("COLLECTION ",embedding_collection)
    VECTOR_SEARCH_INDEX_NAME = "vector_index"

    # Check if java_collection exists
    if embedding_collection_name not in  db.list_collection_names():
        print("Collection doesnot exist so creating new")
        # If java_collection does not exist, create it and populate it from the Java collection
        new_documents = []
        #async for record in collection.find():
        for record in collection.find():
            # Generate embedding content for each record
            page_content = generate_embedding_content(record)
            document = Document(page_content=page_content, metadata={
                "Path Name": record["Path Name"],
                "Current Skill": record["Current Skill"],
                "Path Type": record["Path Type"],
                "Current Proficiency": record["Current Proficiency"],
                "Recommended": record["Recommended"],
                "Prerequisites": record["Prerequisites"],
                "Learning Resources": record["Learning Resources"],
                "What you will learn": record["What you will learn"],
                "Estimated Duration or Course Duration": record["Estimated Duration or Course Duration"],
                "Status" :record["status"],
                "date_added" :current_date,
                "last_updated": current_date
            
            })
            new_documents.append(document)
            print(new_documents[0])
            inserted_count += 1  
    else :
            # If java_collection exists, proceed with "inserted," "updated," and "deleted" logic
            inserted_count, updated_count, deleted_count = 0, 0, 0
            print("Collection  exist looking for updates and insert new")
            new_documents = []
            delete_operations = []

            # Iterate over records in the Java collection for status-based embedding creation
            # async for record in collection.find({"status": {"$in": ["insert", "update"]}}):
            for record in collection.find():
                filter_criteria = {
                    "metadata.Path Name": record["Path Name"],
                    "metadata.Current Skill": record["Current Skill"],
                    "metadata.Current Proficiency": record["Current Proficiency"]
                }
                print("FILTER_CRITER",filter_criteria)
                # Retrieve the existing document for comparison
                existing_embedding =  embedding_collection.find_one(filter_criteria)
                print("EXISITNG:",existing_embedding)
                
                # Generate embedding content
                page_content = generate_embedding_content(record)
                document = Document(page_content=page_content, metadata={
                    "Path Name": record["Path Name"],
                    "Current Skill": record["Current Skill"],
                    "Path Type": record["Path Type"],
                    "Current Proficiency": record["Current Proficiency"],
                    "Recommended": record["Recommended"],
                    "Prerequisites": record["Prerequisites"],
                    "Learning Resources": record["Learning Resources"],
                    "What you will learn": record["What you will learn"],
                    "Estimated Duration or Course Duration": record["Estimated Duration or Course Duration"],
                    "Status" :record["status"],
                    "date_added" :current_date,
                    "last_updated": current_date
                })

                 # Insert logic for new embeddings
                if not existing_embedding:
                    new_documents.append(document)
                    inserted_count += 1
                else:
                    # Compare fields to check if an update is needed
                    update_needed = any(
                        existing_embedding["metadata"].get(field) != record.get(field)
                        for field in [
                            "Path Type", "Recommended", "Prerequisites", "Learning Resources",
                            "What you will learn", "Estimated Duration or Course Duration"
                        ]
                    )

                    if update_needed:
                        print("UPDATE EXIST")
                        # Delete old embedding if necessary
                        delete_operations.append(DeleteOne(filter_criteria))
                        # Add the updated document to the list for embedding creation
                        new_documents.append(document)
                        updated_count += 1

                
                    else:
                        # record["status"] == "nochange":
                        print("*****No changes in embeddings *****")
                        no_change_count += 1

            # Execute delete operations for records marked as "updated"
            if delete_operations:
                delete_result =  embedding_collection.bulk_write(delete_operations)
                deleted_count += delete_result.deleted_count

    # Create or update embeddings for new/updated records
    if new_documents:
        # vectorstore = MongoDBAtlasVectorSearch.from_documents(
        #     documents=new_documents,
        #     embedding=embeddings,
        #     collection=embedding_collection,
        #     index_name=VECTOR_SEARCH_INDEX_NAME
        # )
        # Create embeddings from the data, save to the database and return a connection to MongoDB vCore
        vector_store: AzureCosmosDBVectorSearch = AzureCosmosDBVectorSearch.from_documents(
            documents=new_documents,
            embedding=embeddings,
            collection=embedding_collection,
            index_name=VECTOR_SEARCH_INDEX_NAME,
        )
    print (updated_count, inserted_count,deleted_count, no_change_count)
    # Return summary
    return JSONResponse(content={
        "message": "Vectors created or updated successfully.",
        "inserted": inserted_count,
        "updated": updated_count,
        "deleted": deleted_count,
        "no_change" : no_change_count
    })
# Pydantic model for chat requests
class DirectChatRequest(BaseModel):
    skill_type: str
    query: str
# Pydantic model for chat requests
class RChatRequest(BaseModel):
    skill_type: str
    current_role: str
    proficiency: str 
    target_role: str 
    target_proficiency :str
# Pydantic model for chat requests
# Pydantic model for chat requests
class VChatRequest(BaseModel):
    skill_type: str
    current_role: str
    query:str
    proficiency: str 
    target_role: str 
    target_proficiency :str
class ChatRequest(BaseModel):
    skill_type: str
    skill: str
    query: str
    proficiency: str 
    path_type: str 
class searchRequest(BaseModel):
    skill_type: str
    skill: str
    proficiency: str 
    path_type: str 
@app.post("/search-vectors/")
async def search_vectors(request:ChatRequest):
    #client = MongoClient(os.environ["CMONGO_URI"])
    client = MongoClient(CMONGO_URI)
    skill_type = request.skill_type
    skill = request.skill
    query= request.query
    db = client.skill_data
    collection = db[skill]
     
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
    print("Query",query)
    print("Skill",skill)
    proficiency = request.proficiency
    path_type = request.path_type
    collection = db[f"{skill_type}_collection"]

    pre_filter = {
    "Current Skill": {skill},
    "Path Type": {path_type},   
    "Current Proficiency": {proficiency}
     }  
    vectorstore = MongoDBAtlasVectorSearch(embedding=embeddings, collection=collection,index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME)
    results = vectorstore.similarity_search(query,k=10)
    print("Results",results[0])
    # Filter results based on proficiency and path_type if provided
    # filtered_results = []
    # for result in results:
    #     if proficiency and result.metadata.get("Current Proficiency") != proficiency:
    #         continue
    #     if path_type and result.metadata.get("Path Type") != path_type:
    #         continue
    #     filtered_results.append(result)

    return {"results": results}




# Convert MongoDB dictionaries to LangChain Document objects
def prepare_documents_for_llm(records):
        documents = []
        for record in records:
            content = record.get("textContent", "")
            # Extract metadata fields from the nested 'metadata' object
            metadata_obj = record.get("metadata", {})
            metadata = {
                "Current Skill": metadata_obj.get("Current Skill", ""),
                "Path Type": metadata_obj.get("Path Type", ""),
                "Current Proficiency": metadata_obj.get("Current Proficiency", ""),
                "Path Name": metadata_obj.get("Path Name", ""),
                "Prerequisites": metadata_obj.get("Prerequisites", ""),
                "Learning Resources": metadata_obj.get("Learning Resources", ""),
                "What you will learn": metadata_obj.get("What you will learn", ""),
                "Estimated Duration or Course Duration": metadata_obj.get("Estimated Duration or Course Duration", ""),

            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

@app.post("/direct_search-vectors/")
async def search_vectors(request:searchRequest):
    #client = MongoClient(os.environ["CMONGO_URI"])
    client = MongoClient(CMONGO_URI)
    skill_type = request.skill_type
    skill = request.skill
    
    db = client.skill_data
    collection = db[skill]
     
    VECTOR_SEARCH_INDEX_NAME = "vector_index"
   
    print("Skill",skill)
    proficiency = request.proficiency
    path_type = request.path_type
    print("proficiency",proficiency)
    print("path_type",path_type)

    collection = db[f"{skill_type}_collection"]

    pre_filter = {
    "Current Skill": {skill},
    "Path Type": {path_type},   
    "Current Proficiency": {proficiency}
     }  
    
    # Function to filter records directly from MongoDB
    def filter_records(skill, proficiency, path_type):
            filter_criteria = {
                "metadata.Current Skill": skill,
                "metadata.Path Type": path_type,
                "metadata.Current Proficiency": proficiency
            }
            # Retrieve all fields in the document
            projection = {
                    "metadata.Current Skill": 1,
                    "metadata.Path Type": 1,
                    "metadata.Current Proficiency": 1,
                    "metadata.Recommended": 1,
                    "metadata.Path Name": 1,
                    "metadata.Prerequisites": 1,
                    "metadata.Learning Resources": 1,
                    "metadata.What you will learn": 1,
                    "metadata.Estimated Duration or Course Duration": 1,
                    "textContent":1
            }
            results = collection.find(filter_criteria, projection).to_list(length=10)
            return results
    # Combine MongoDB filtering with LangChain querying
    def combine_filter_and_query(skill, proficiency, path_type):
        # Step 1: Get filtered documents from MongoDB
        filtered_documents = filter_records(skill, proficiency, path_type)
    
        # Step 2: Convert filtered documents to Document format for LLM
        prepared_documents = prepare_documents_for_llm(filtered_documents)
             
        return prepared_documents 
    
    # Get the answer from the combined process
    prepared_documents = combine_filter_and_query(skill, proficiency, path_type)
    print("Length",len(prepared_documents))

    if len(prepared_documents) > 0 :
    

         print("Results",prepared_documents[0])
         
    else :
         
          prepared_documents = []
        
          content = " "
          metadata = {
                "Result": "No information "
            }
          prepared_documents.append(Document(page_content=content, metadata=metadata))
         
         

    return {"results": prepared_documents}



@app.post("/chat/")
async def search_vectors(request:ChatRequest):
    #Mclient = MongoClient(os.environ["CMONGO_URI"])
    Mclient = MongoClient(CMONGO_URI)
    client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
    def llm_call(prompt) :
        completion = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages= [
            {
            "role": "user",
            "content": prompt



            }],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        #print(completion.to_json())
    
        return completion.choices[0].message.content.strip()
    skill_type = request.skill_type
    print("SKILL TYPE",skill_type)
    skill = request.skill
    query= request.query
    db = Mclient.skill_data
    collection = db[skill_type]
   
     
    VECTOR_SEARCH_INDEX_NAME = "vector_index"
    print("Query",query)
    print("Skill",skill)
    proficiency = request.proficiency
    path_type = request.path_type
    print("proficiency",proficiency)
    print("path_type",path_type)

    collection = db[f"{skill_type}_collection"]
    print("COLLECTION.",collection)

    pre_filter = {
    "Current Skill": {skill},
    "Path Type": {path_type},   
    "Current Proficiency": {proficiency}
     }  
    
    # Function to filter records directly from MongoDB
    def filter_records(skill, proficiency, path_type):
            filter_criteria = {
                "metadata.Current Skill": skill,
                "metadata.Path Type": path_type,
                "metadata.Current Proficiency": proficiency
            }
            # Retrieve all fields in the document
            projection = {
                "metadata.Current Skill": 1,
                "metadata.Path Type": 1,
                "metadata.Current Proficiency": 1,
                "metadata.Path Name": 1,
                "textContent":1
            }
            results = collection.find(filter_criteria, projection).to_list(length=10)
            return results
    # Combine MongoDB filtering with LangChain querying
    def combine_filter_and_query(skill, proficiency, path_type):
        # Step 1: Get filtered documents from MongoDB
        filtered_documents = filter_records(skill, proficiency, path_type)
        #print("FILTER DOC",filtered_documents)
    
        # Step 2: Convert filtered documents to Document format for LLM
        prepared_documents = prepare_documents_for_llm(filtered_documents)
             
        return prepared_documents 
    
    # Get the answer from the combined process
    prepared_documents = combine_filter_and_query(skill, proficiency, path_type)
    print("Length",len(prepared_documents))
    #print("DOCs raw",prepared_documents)
    answer = "No information present at this moment "
    if len(prepared_documents) > 0 :
        context= prepared_documents
        prompt = f"""
        You are a helpful assistant. Based on the following Context, answer the question ,dont answer any information from internet or
        any information not exist in specified Context if there is no answer just say 'No information present at this moment':

        Context: {context}
        Question: {query}
        """
        print("Call to llm")
        #print("PROMPT",prompt)
        answer = llm_call(prompt)
        print("Results",answer)
        print("Call end to llm")

    return {"results": answer}

########################### ROLE BASED DATA ######################


@app.post("/rolebased-upload-csv/")
async def upload_csv(skill_type: str = Form(...), file: UploadFile = File(...)):
   

    # Read the CSV file into a DataFrame
    df = pd.read_csv(io.StringIO((await file.read()).decode("utf-8")))
    

    # Define the collection based on the skill type
    collection = db[f"{skill_type}_role"]
    #collection = db[f"{skill}_collection"]
    # Define the current date for insertion and update timestamps
    current_date = datetime.utcnow()

    # Track inserted, updated, and no change counts
    inserted_count = 0
    updated_count = 0
    no_change_count = 0

    # Prepare bulk operations
    bulk_operations = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Define filter criteria based on relevant fields
        filter_criteria = {
            "Role": row["Role"],
            "Proficiency Level": row["Proficiency Level"]
            
        }

        # Retrieve the existing document for comparison
        existing_doc = await collection.find_one(filter_criteria)

        if existing_doc is not None:
            # Compare fields to check for changes
            update_fields = {}
            for field in ["Topics", "Courses"]:
                if existing_doc.get(field) != row.get(field):
                    update_fields[field] = row.get(field)

            if update_fields:  # If there are fields to update
                update_fields["last_updated"] = current_date  # Add last_updated
                update_fields["status"] = "update"  # Set status to update
                bulk_operations.append(UpdateOne(
                    filter_criteria,
                    {
                        "$set": update_fields
                    }
                ))
                updated_count += 1  # Increment updated count
            else:
                # If there are no changes, update status to nochange
                bulk_operations.append(UpdateOne(
                    filter_criteria,
                    {
                        "$set": {"status": "nochange"}
                    }
                ))
                no_change_count += 1  # Increment no change count
        else:
            # Document does not exist; prepare an insert operation
            insert_data = {
                "Role": row["Role"],
                "Proficiency Level": row["Proficiency Level"],
                
                "Topics": row.get("Topics"),
                "Courses": row.get("Courses"),
                "last_updated": current_date,
                "status": "insert"  # Set status to insert
            }
            bulk_operations.append(UpdateOne(
                filter_criteria,
                {
                    "$set": insert_data,
                    "$setOnInsert": {"date_added": current_date}
                },
                upsert=True  # Ensure the document gets inserted if it doesn't exist
            ))
            # inserted_count += 1  # Increment inserted count

    # Execute bulk write if there are any operations to perform
    if bulk_operations:
        result = await collection.bulk_write(bulk_operations)
        # Increment inserted count for the new documents that were inserted
        inserted_count += len(result.upserted_ids)

    # Return the counts of the operations performed
    return {
        "status": "success",
        "inserted": inserted_count,
        "updated": updated_count,
        "no_change": no_change_count
    }
# Helper function to convert ObjectId to string and handle NaN/Infinity values
def clean_data_for_json(data):
    cleaned_data = []
    for item in data:
        # Convert ObjectId to string
        item["_id"] = str(item["_id"])

        # Replace non-JSON-compliant values
        for key, value in item.items():
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                item[key] = None  # Replace NaN or Inf with None
        cleaned_data.append(item)
    
    return cleaned_data

@app.get("/rolebased-retrieve-data/")
async def retrieve_data(skill_type: str):
    if skill_type not in skills:
        return {"error": "Invalid skill type"}
    
    collection = db[f"{skill_type}_role"]
    data = await collection.find().to_list(length=100)  # Limit the results
    
    # Clean data for JSON serialization
    data = clean_data_for_json(data)
    
    return {"data": data}

def rolebased_generate_embedding_content(record):
    return (
        f"if your current role is {record['Role']} and proficiency level is  {record['Proficiency Level']} "
        f" recommended topics {record['Topics']} with Courses {record['Courses']} for tTransition Guidance Between Roles"
    )

@app.post("/rolebased-create-vectors-crud-new/")
async def create_vectors(request: VectorCreationRequest):
    #client = MongoClient(os.environ["MONGO_URI"])
    client = MongoClient(CMONGO_URI)
    #client = AsyncIOMotorClient(os.environ["MONGO_URI"])
    db = client.skill_data
    skill=request.skill_type
    #collection = db[skill]
    data = []
    #embedding_collection = db[f"{skill}_collection"]
    VECTOR_SEARCH_INDEX_NAME = "vector_index"
    # Define the current date for insertion and update timestamps
    current_date = datetime.utcnow()

    # Track inserted, updated, and no change counts
    inserted_count = 0
    updated_count = 0
    deleted_count = 0
    no_change_count = 0

    db = client.skill_data
   
    embedding_collection_name = f"{skill}_rolecollection"
    collection = db[f"{skill}_role"]
    embedding_collection = db[f"{skill}_rolecollection"]
    VECTOR_SEARCH_INDEX_NAME = "vector_index"

    # Check if java_collection exists
    if embedding_collection_name not in  db.list_collection_names():
        print("Collection doesnot exist so creating new")
        # If java_collection does not exist, create it and populate it from the Java collection
        new_documents = []
        #async for record in collection.find():
        for record in collection.find():
            # Generate embedding content for each record
            # Generate embedding content
            page_content = rolebased_generate_embedding_content(record)
            document = Document(page_content=page_content, metadata={
                "Role": record["Role"],
                "Proficiency Level": record["Proficiency Level"],
                "Topics": record['Topics'],
                "Courses": record['Courses'],
                "Status" :record["status"],
                "date_added" :current_date,
                "last_updated": current_date
                
            })
            new_documents.append(document)
            print(new_documents[0])
            inserted_count += 1  
    else :
            # If java_collection exists, proceed with "inserted," "updated," and "deleted" logic
            inserted_count, updated_count, deleted_count = 0, 0, 0
            print("Collection  exist looking for updates and insert new")
            new_documents = []
            delete_operations = []

            # Iterate over records in the Java collection for status-based embedding creation
            # async for record in collection.find({"status": {"$in": ["insert", "update"]}}):
            for record in collection.find():
                filter_criteria = {
                    "metadata.Role": record["Role"],
                    "metadata.Proficiency Level": record["Proficiency Level"]
                    
                }
                # Retrieve the existing document for comparison
                existing_embedding =  embedding_collection.find_one(filter_criteria)
                
                # Generate embedding content
                page_content = rolebased_generate_embedding_content(record)
                document = Document(page_content=page_content, metadata={
                    "Role": record["Role"],
                    "Proficiency Level": record["Proficiency Level"],
                    "Topics": record['Topics'],
                    "Courses": record['Courses'],
                    "Status" :record["status"],
                    "date_added" :current_date,
                    "last_updated": current_date
                })

                 # Insert logic for new embeddings
                if not existing_embedding:
                    new_documents.append(document)
                    inserted_count += 1
                else:
                    # Compare fields to check if an update is needed
                    update_needed = any(
                        existing_embedding["metadata"].get(field) != record.get(field) for field in [
                            "Role", "Proficiency Level", "Topics", "Courses"]
                    )

                    if update_needed:
                        # Delete old embedding if necessary
                        delete_operations.append(DeleteOne(filter_criteria))
                        # Add the updated document to the list for embedding creation
                        new_documents.append(document)
                        updated_count += 1

                
                    else:
                        # record["status"] == "nochange":
                        print("*****No changes in embeddings *****")
                        no_change_count += 1

            # Execute delete operations for records marked as "updated"
            if delete_operations:
                delete_result =  embedding_collection.bulk_write(delete_operations)
                deleted_count += delete_result.deleted_count

    # Create or update embeddings for new/updated records
    if new_documents:
        # vectorstore = MongoDBAtlasVectorSearch.from_documents(
        #     documents=new_documents,
        #     embedding=embeddings,
        #     collection=embedding_collection,
        #     index_name=VECTOR_SEARCH_INDEX_NAME
        # )
        # Create embeddings from the data, save to the database and return a connection to MongoDB vCore
        vector_store: AzureCosmosDBVectorSearch = AzureCosmosDBVectorSearch.from_documents(
            documents=new_documents,
            embedding=embeddings,
            collection=embedding_collection,
            index_name=VECTOR_SEARCH_INDEX_NAME,
        )
    print (updated_count, inserted_count,deleted_count, no_change_count)
    # Return summary
    return JSONResponse(content={
        "message": "Vectors created or updated successfully.",
        "inserted": inserted_count,
        "updated": updated_count,
        "deleted": deleted_count,
        "no_change" : no_change_count
    })






@app.get("/rolebased-retrieve-data-vector/")
async def retrieve_data_vector(skill_type: str):
    print("SkillTYpe",skill_type)
    # client = MongoClient(os.environ["MONGO_URI"])
    client = AsyncIOMotorClient(CMONGO_URI)
    # if skill_type not in skills:
    #     return {"error": "Invalid skill type"}
    db = client.skill_data
    collection =  db[f"{skill_type}_rolecollection"]
    data = await collection.find().to_list(length=100)  # Limit the results
    
    # Clean data for JSON serialization
    data = clean_data_for_json(data)
    
    return {"data": data}


@app.post("/rolebased-chat/")
async def search_vectors(request:DirectChatRequest):
    
    Mclient = MongoClient(CMONGO_URI)

    skill_type = request.skill_type
    query= request.query
    

    client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
     
    def llm_call(prompt) :
        completion = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages= [
            {
            "role": "user",
            "content": prompt



            }],
            max_tokens=1000,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        #print(completion.to_json())
    
        return completion.choices[0].message.content.strip()
    
    
    db = Mclient.skill_data
    collection=db[f"{skill_type}_rolecollection"]
     
    VECTOR_SEARCH_INDEX_NAME = "vector_index"
    print("Query",query)
    print("Skill Type",skill_type)
    

    # Fetch data from MongoDB
    try:
        documents = collection.find(
            {},
            {
                "metadata.Role": 1,
                "metadata.Proficiency Level": 1,
                "metadata.Topics": 1,
                "metadata.Courses": 1,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # Prepare context from documents
    context_list = [
        f"Role: {doc.get('metadata', {}).get('Role', '')}, "
        f"Proficiency: {doc.get('metadata', {}).get('Proficiency Level', '')}, "
        f"Topics: {doc.get('metadata', {}).get('Topics', '')}, "
        f"Courses: {doc.get('metadata', {}).get('Courses', '')}"
        for doc in documents
    ]
    context = "\n".join(context_list)

    if not context:
        raise HTTPException(status_code=404, detail="No data found in the collection.")

    # Construct the prompt for Azure OpenAI
    prompt = (
        f"Your are an expert in providing Transition Guidance Between Roles based on provided data.\n"
        f"Make the best Roadmaps for Comprehensive Learning based on the provided context. "
        f"Use context data only; do not use external knowledge.\n\n"
        f"Respond in a professional tone. Always provide high-level stages or hops (if applicable) first, "
        f"followed by detailed steps.\n\n"
        f"If no information is available in the context, respond with: 'No information present at this moment.'\n\n"
        f"Example:\n"
        f"If transitioning from 'Java Developer' to 'Java Team Lead', show hops like:\n"
        f"Java Developer -> Java Senior Developer -> Java Team Lead\n"
        f"Then provide details for each stage.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    print("PROMPT:\n", prompt)

    # Get the response from Azure OpenAI
    try:
        answer = llm_call(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    print("Results:\n", answer)
    return {"results": answer}

@app.post("/rolebased-Direct_chat/")
async def search_vectors(request:RChatRequest):
    
    Mclient = MongoClient(CMONGO_URI)

    skill_type = request.skill_type
    current_role = request.current_role
    #query= request.query
    proficiency = request.proficiency
    target_role = request.target_role
    target_proficiency = request.target_proficiency

    client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
     
    def llm_call(prompt) :
        completion = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages= [
            {
            "role": "user",
            "content": prompt



            }],
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        #print(completion.to_json())
    
        return completion.choices[0].message.content.strip()
    
    
    db = Mclient.skill_data
    collection=db[f"{skill_type}_rolecollection"]
     
    VECTOR_SEARCH_INDEX_NAME = "vector_index"
    #print("Query",query)
    print("Skill Type",skill_type)
    print("Current Role",current_role)
    
    print("proficiency",proficiency)
    print("Target Role",target_role)
    if current_role == target_role:

        # Define the filter for multiple roles
        #mquery = {"Role": {"$in": ["Java Developer", "Java Team Lead"]}}
        mquery = {"metadata.Role": {"$in": [current_role, target_role]}}
    else :

        mquery = {}
    print("MQUERY:",mquery)
    # Retrieve only selected fields from each document in the MongoDB collection
    documents = collection.find(mquery, {"metadata.Role": 1, "metadata.Proficiency Level": 1, "metadata.Topics": 1, "metadata.Courses": 1})
    print("DOCUMNETS:",documents)
    # Combine the content from the selected fields into a single context string
    context_list = [
        f"Role: {doc.get('metadata', {}).get('Role', '')}, "
        f"Proficiency: {doc.get('metadata', {}).get('Proficiency Level', '')}, "
        f"Topics: {doc.get('metadata', {}).get('Topics', '')}, "
        f"Courses: {doc.get('metadata', {}).get('Courses', '')}"
        for doc in documents
    ]
    context = "\n".join(context_list)
    print("CONTEXT",context)
    # Check if there is any content in the context
    if not context:
        raise HTTPException(status_code=404, detail="No data found in the collection.")
    

    answer = "No information present at this moment "

    query=f"""I am currently a {current_role}  which is my 'CURRENT ROLE' with an {proficiency} level of proficiency and 
    aim to progress toward a {target_role} 'TARGET ROLE' with an {target_proficiency} level of proficiency.
    Could you recommend a structured transition path ,courses and topics for 
    advancing my skills and responsibilities based on asku"""
    print("Query",query)  
    # Define the prompt for Azure OpenAI
    prompt = f""" Your are expert in helping Transition Guidance Between Roles  based on provided roles,
            Make best Roadmaps for Comprehensive Learning based on 'CURRENT ROLE' -> 'TARGET ROLE' .
            Provide answer based on provided context data only.
        
        Always First show complete high level hops or stages 'CURRENT ROLE' -> 'TARGET ROLE' 
        after that provide further intermidiate details  and respond in professional tone.
        
        if you cannot provide information from given context just say "No information present at this moment"
        
        example 'if i am Java Developer and want to became Java Team Lead' 
        First step you need to consider hops or intermidate stages then go for details
        
        Java Developer ----> Java Senior Developer  ---> JAva Team Lead... 
        First show high level hops or stages then go for details about each stages what to be learned.
        
        
        
        
        context:\n\nContext:\n{context}\n\n
        Question: {query}\nAnswer:"""


    print("PROMPT:",prompt)   
    answer = llm_call(prompt)

    print("Results",answer)
    print("Call end to llm")
    return {"results": answer}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
