from src.collection.google_scholar import GoogleScholarScraper

def test_scraper():
    scraper = GoogleScholarScraper()
    print("Testing scraper...")
    try:
        # Test with a simple query
        results = scraper.search("machine learning", num_results=1)
        print(f"Results found: {len(results)}")
        if results:
            print(results[0])
        else:
            print("No results. Checking raw response...")
            # We might need to modify the class to expose raw response for debugging
            # For now, let's just see if it runs without error
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_scraper()
