"""
Comprehensive Backend API Testing Script
Tests all endpoints and functionality
"""

import requests
import json
import time
from typing import Dict, Any


API_BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_result(test_name: str, success: bool, response: Dict = None, error: str = None):
    """Print test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"\n{status} - {test_name}")
    
    if response:
        print(f"Response: {json.dumps(response, indent=2)}")
    
    if error:
        print(f"Error: {error}")


def test_health_check():
    """Test 1: Health Check Endpoint"""
    print_section("TEST 1: HEALTH CHECK")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_result("Health Check", True, data)
            return True
        else:
            print_result("Health Check", False, error=f"Status code: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print_result("Health Check", False, error="Cannot connect to API. Is the server running?")
        print("\n💡 Start the server with: python scripts/run_backend.py")
        return False
    
    except Exception as e:
        print_result("Health Check", False, error=str(e))
        return False


def test_root_endpoint():
    """Test 2: Root Endpoint"""
    print_section("TEST 2: ROOT ENDPOINT")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        
        if response.status_code == 200:
            data = response.json()
            print_result("Root Endpoint", True, data)
            return True
        else:
            print_result("Root Endpoint", False, error=f"Status code: {response.status_code}")
            return False
    
    except Exception as e:
        print_result("Root Endpoint", False, error=str(e))
        return False


def test_list_charities():
    """Test 3: List Charities Endpoint"""
    print_section("TEST 3: LIST CHARITIES")
    
    try:
        response = requests.get(f"{API_BASE_URL}/scrape/charities")
        
        if response.status_code == 200:
            data = response.json()
            print_result("List Charities", True, data)
            
            charities = data.get('charities', [])
            if charities:
                print(f"\n📚 Found {len(charities)} indexed charities:")
                for charity in charities:
                    if isinstance(charity, dict):
                        print(f"  - {charity.get('name', 'Unknown')}: {charity.get('chunk_count', 0)} chunks")
                    else:
                        print(f"  - {charity}")
            else:
                print("\n⚠️  No charities indexed yet. Run: python scripts/index_sample_data.py")
            
            return True, charities
        else:
            print_result("List Charities", False, error=f"Status code: {response.status_code}")
            return False, []
    
    except Exception as e:
        print_result("List Charities", False, error=str(e))
        return False, []


def test_chat_basic():
    """Test 4: Basic Chat Query"""
    print_section("TEST 4: BASIC CHAT QUERY")
    
    try:
        payload = {
            "query": "What programs does Red Cross offer?",
            "charity_name": "Red Cross",
            "top_k": 5
        }
        
        print(f"\n📤 Sending request:")
        print(f"   Query: {payload['query']}")
        print(f"   Charity: {payload['charity_name']}")
        print(f"   Top-K: {payload['top_k']}")
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json=payload,
            timeout=30
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ Response received in {duration:.2f}s")
            print(f"\n📊 Results:")
            print(f"   Status: {data.get('status')}")
            print(f"   Retrieved Chunks: {data.get('retrieved_chunks', 0)}")
            print(f"   Processing Time: {data.get('processing_time', 0):.2f}s")
            
            print(f"\n💬 Answer:")
            print(f"   {data.get('response', 'No response')}")
            
            sources = data.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources[:3], 1):  # Show first 3
                    print(f"   {i}. Similarity: {source.get('similarity', 0):.2f}")
                    print(f"      Text: {source.get('text', '')[:100]}...")
            
            return True, data
        else:
            error_data = response.json() if response.text else {}
            print_result("Basic Chat Query", False, error=f"Status: {response.status_code}, {error_data}")
            return False, None
    
    except Exception as e:
        print_result("Basic Chat Query", False, error=str(e))
        return False, None


def test_chat_without_charity_filter():
    """Test 5: Chat Without Charity Filter"""
    print_section("TEST 5: CHAT WITHOUT CHARITY FILTER")
    
    try:
        payload = {
            "query": "What healthcare services are available?",
            "top_k": 3
        }
        
        print(f"\n📤 Sending request:")
        print(f"   Query: {payload['query']}")
        print(f"   Charity: (All charities)")
        
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✅ Response:")
            print(f"   Retrieved Chunks: {data.get('retrieved_chunks', 0)}")
            print(f"   Answer: {data.get('response', 'No response')[:200]}...")
            
            return True
        else:
            print_result("Chat Without Filter", False, error=f"Status: {response.status_code}")
            return False
    
    except Exception as e:
        print_result("Chat Without Filter", False, error=str(e))
        return False


def test_chat_multiple_queries():
    """Test 6: Multiple Sequential Queries"""
    print_section("TEST 6: MULTIPLE QUERIES")
    
    queries = [
        {"query": "What is Red Cross mission?", "charity": "Red Cross"},
        {"query": "How does UNICEF help children?", "charity": "UNICEF"},
        {"query": "Tell me about Doctors Without Borders", "charity": "Doctors Without Borders"}
    ]
    
    results = []
    
    for i, test_query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Question: {test_query['query']}")
        print(f"Charity: {test_query['charity']}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/",
                json={
                    "query": test_query['query'],
                    "charity_name": test_query['charity'],
                    "top_k": 5
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success - Retrieved {data.get('retrieved_chunks', 0)} chunks")
                print(f"Answer: {data.get('response', '')[:150]}...")
                results.append(True)
            else:
                print(f"❌ Failed - Status: {response.status_code}")
                results.append(False)
        
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(False)
        
        time.sleep(1)  # Small delay between requests
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Success Rate: {success_rate:.0f}% ({sum(results)}/{len(results)} passed)")
    
    return all(results)


def test_chat_edge_cases():
    """Test 7: Edge Cases"""
    print_section("TEST 7: EDGE CASES")
    
    edge_cases = [
        {
            "name": "Empty query",
            "payload": {"query": "", "charity_name": "Red Cross"},
            "should_fail": True
        },
        {
            "name": "Very long query",
            "payload": {"query": "What " + "really " * 100 + "long query", "charity_name": "Red Cross"},
            "should_fail": True
        },
        {
            "name": "Non-existent charity",
            "payload": {"query": "What do you do?", "charity_name": "Fake Charity 123"},
            "should_fail": False  # Should return no results
        },
        {
            "name": "Invalid top_k (too high)",
            "payload": {"query": "Help", "charity_name": "Red Cross", "top_k": 100},
            "should_fail": True
        }
    ]
    
    results = []
    
    for test_case in edge_cases:
        print(f"\n--- {test_case['name']} ---")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/",
                json=test_case['payload'],
                timeout=10
            )
            
            if test_case['should_fail']:
                if response.status_code >= 400:
                    print(f"✅ Correctly rejected - Status: {response.status_code}")
                    results.append(True)
                else:
                    print(f"❌ Should have failed but passed")
                    results.append(False)
            else:
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Handled gracefully - Retrieved: {data.get('retrieved_chunks', 0)} chunks")
                    results.append(True)
                else:
                    print(f"❌ Failed unexpectedly - Status: {response.status_code}")
                    results.append(False)
        
        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append(False)
    
    return all(results)


def test_session_management():
    """Test 8: Session Management"""
    print_section("TEST 8: SESSION MANAGEMENT")
    
    try:
        # Create a session with multiple messages
        session_id = f"test_session_{int(time.time())}"
        
        print(f"\n📝 Creating session: {session_id}")
        
        # Send first message
        response1 = requests.post(
            f"{API_BASE_URL}/chat/",
            json={
                "query": "What is Red Cross?",
                "charity_name": "Red Cross",
                "session_id": session_id
            }
        )
        
        # Send second message
        response2 = requests.post(
            f"{API_BASE_URL}/chat/",
            json={
                "query": "How can I volunteer?",
                "charity_name": "Red Cross",
                "session_id": session_id
            }
        )
        
        # Get history
        history_response = requests.get(f"{API_BASE_URL}/chat/history/{session_id}")
        
        if history_response.status_code == 200:
            history = history_response.json()
            print(f"✅ Session created successfully")
            print(f"   Messages in history: {history.get('message_count', 0)}")
            
            # Clear history
            clear_response = requests.delete(f"{API_BASE_URL}/chat/history/{session_id}")
            
            if clear_response.status_code == 200:
                print(f"✅ History cleared successfully")
                return True
            else:
                print(f"❌ Failed to clear history")
                return False
        else:
            print(f"❌ Failed to retrieve history")
            return False
    
    except Exception as e:
        print_result("Session Management", False, error=str(e))
        return False


def test_api_documentation():
    """Test 9: API Documentation"""
    print_section("TEST 9: API DOCUMENTATION")
    
    try:
        # Test OpenAPI docs
        docs_response = requests.get(f"{API_BASE_URL}/docs")
        redoc_response = requests.get(f"{API_BASE_URL}/redoc")
        openapi_response = requests.get(f"{API_BASE_URL}/openapi.json")
        
        docs_available = docs_response.status_code == 200
        redoc_available = redoc_response.status_code == 200
        openapi_available = openapi_response.status_code == 200
        
        print(f"\n📚 Documentation Endpoints:")
        print(f"   Swagger UI (/docs): {'✅ Available' if docs_available else '❌ Not available'}")
        print(f"   ReDoc (/redoc): {'✅ Available' if redoc_available else '❌ Not available'}")
        print(f"   OpenAPI Spec (/openapi.json): {'✅ Available' if openapi_available else '❌ Not available'}")
        
        if docs_available:
            print(f"\n💡 View interactive docs at: {API_BASE_URL}/docs")
        
        return docs_available and openapi_available
    
    except Exception as e:
        print_result("API Documentation", False, error=str(e))
        return False


def run_all_tests():
    """Run all backend tests"""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "BACKEND API TEST SUITE" + " "*37 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    results = []
    
    # Test 1: Health Check (critical)
    health_ok = test_health_check()
    results.append(("Health Check", health_ok))
    
    if not health_ok:
        print("\n❌ CRITICAL: API server is not running or not responding")
        print("   Please start the server first: python scripts/run_backend.py")
        return
    
    # Test 2: Root Endpoint
    results.append(("Root Endpoint", test_root_endpoint()))
    
    # Test 3: List Charities
    list_ok, charities = test_list_charities()
    results.append(("List Charities", list_ok))
    
    if not charities:
        print("\n⚠️  WARNING: No charities indexed")
        print("   Some tests may fail. Run: python scripts/index_sample_data.py")
        print("\n   Continue with tests? (y/n): ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("\n   Exiting tests. Please index data first.")
            return
    
    # Test 4: Basic Chat
    results.append(("Basic Chat Query", test_chat_basic()[0]))
    
    # Test 5: Chat Without Filter
    results.append(("Chat Without Filter", test_chat_without_charity_filter()))
    
    # Test 6: Multiple Queries
    results.append(("Multiple Queries", test_chat_multiple_queries()))
    
    # Test 7: Edge Cases
    results.append(("Edge Cases", test_chat_edge_cases()))
    
    # Test 8: Session Management
    results.append(("Session Management", test_session_management()))
    
    # Test 9: API Documentation
    results.append(("API Documentation", test_api_documentation()))
    


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
