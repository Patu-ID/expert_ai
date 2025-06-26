#!/usr/bin/env python3
"""
Test script for the new document endpoints in ExpertORT Agent API.
This script demonstrates file indexing and document querying capabilities.
"""

import requests
import json
import os

def test_document_endpoints():
    """Test the document API endpoints."""
    base_url = "http://localhost:8081"
    
    print("📄 Testing ExpertORT Document API Endpoints")
    print("=" * 60)
    
    # Test health check first
    print("\n🏥 Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API is healthy")
            print(f"   Document indexing available: {health.get('document_indexing_available', False)}")
            print(f"   Document retrieval available: {health.get('document_retrieval_available', False)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the server is running on port 8081")
        print("   Run: python app.py")
        return
    
    # Test 1: Query existing documents
    print("\n1️⃣ Testing document query endpoint...")
    query_data = {
        "query": "What is the Transformer architecture?",
        "k": 5,
        "top_p": 3,
        "min_score": 0.1
    }
    
    try:
        response = requests.post(
            f"{base_url}/v1/documents/query",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Query successful!")
            print(f"📊 Query: {result['query']}")
            print(f"📊 Status: {result['status']}")
            print(f"📊 Total candidates: {result['total_candidates']}")
            print(f"📊 Returned results: {result['returned_results']}")
            
            if result['results']:
                print(f"📄 First result preview:")
                first_result = result['results'][0]
                print(f"   - Document: {first_result['document_name']}")
                print(f"   - Score: {first_result['score']:.4f}")
                print(f"   - Content: {first_result['content'][:100]}...")
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Query request failed: {e}")
    
    # Test 2: Get index statistics
    print("\n2️⃣ Testing index statistics endpoint...")
    try:
        response = requests.get(f"{base_url}/v1/documents/index/rag_documents/stats")
        
        if response.status_code == 200:
            stats = response.json()
            print("✅ Index statistics retrieved!")
            print(f"📊 Index name: {stats.get('index_name')}")
            print(f"📊 Total documents: {stats.get('total_documents')}")
            print(f"📊 Unique documents: {stats.get('unique_document_count')}")
            print(f"📊 Document names: {stats.get('unique_document_names', [])}")
        else:
            print(f"❌ Statistics failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Statistics request failed: {e}")
    
    # Test 3: Search by document name
    print("\n3️⃣ Testing document name search endpoint...")
    try:
        response = requests.get(
            f"{base_url}/v1/documents/search/Attention Is All You Need",
            params={"k": 3}
        )
        
        if response.status_code == 200:
            search_result = response.json()
            print("✅ Document search successful!")
            print(f"📊 Document: {search_result['document_name']}")
            print(f"📊 Total chunks: {search_result['total_chunks']}")
            
            if search_result['chunks']:
                print(f"📄 First chunk preview:")
                first_chunk = search_result['chunks'][0]
                print(f"   - Chunk ID: {first_chunk['chunk_id']}")
                print(f"   - Content length: {first_chunk['content_length']}")
                print(f"   - Content: {first_chunk['content'][:100]}...")
        else:
            print(f"❌ Document search failed: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Document search request failed: {e}")
    
    # Test 4: File indexing (commented out as it requires a file)
    print("\n4️⃣ File indexing endpoint info...")
    print("📄 To test file indexing, use the following curl command:")
    print(f'curl -X POST "{base_url}/v1/documents/index" \\')
    print('     -F "file=@your_document.pdf" \\')
    print('     -F "index_name=test_index" \\')
    print('     -F "chunk_size=500" \\')
    print('     -F "chunk_overlap=50"')
    
    print("\n" + "=" * 60)
    print("🎉 Document API testing completed!")
    print("\n📚 Available endpoints:")
    print("   POST /v1/documents/index - Index PDF files")
    print("   POST /v1/documents/query - Query documents with semantic search")
    print("   GET  /v1/documents/index/{index_name}/stats - Get index statistics")
    print("   GET  /v1/documents/search/{document_name} - Search by document name")

if __name__ == "__main__":
    test_document_endpoints()
