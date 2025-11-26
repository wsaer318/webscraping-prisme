from src.collection.crossref_api import CrossrefScraper
import json

def test_crossref():
    scraper = CrossrefScraper()
    print("Testing Crossref Search...")
    results = scraper.search("machine learning", max_results=5)
    
    print(f"Found {len(results)} results.")
    
    if results:
        print("First result:")
        print(json.dumps(results[0], indent=2))
        
        # Verify keys
        required_keys = ['title', 'authors', 'year', 'doi', 'source']
        for key in required_keys:
            if key not in results[0]:
                print(f"WARNING: Missing key {key}")
    else:
        print("No results found.")

if __name__ == "__main__":
    test_crossref()
