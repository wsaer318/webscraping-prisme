#!/usr/bin/env python3
#_*-_ coding: utf-8 _*_


###  Leonel Version
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup  
import json
import os
import time
import random


url =  'https://scholar.google.com/scholar?q='
subjet_search = "proximal policy optimization ppo"
encoded_search = urllib.parse.quote_plus(subjet_search)
results_per_page = '&num=5'
full_url = url + encoded_search + results_per_page
print(full_url)
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
headers = {'User-Agent': user_agent}

logs_json = []
avoid_duplicates = []

file_name = 'leonel/results_google_scholar.json'

def get_abstract(article_url):
    # Function to fetch and parse the article page to extract the abstract
    req = urllib.request.Request(article_url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            page_content = response.read()
            soup = BeautifulSoup(page_content, 'html.parser')
            abstract_tag = soup.find('div', class_='abstract')  # Example class name
            print("Abstract "+ str(abstract_tag))
            abstract = abstract_tag.get_text(strip=True) if abstract_tag else "Abstract not found"
            return abstract
    except Exception as e:
        print(f"Error fetching abstract: {e}")
        return "Abstract not found"

def get_article(article): 
    # Function to fetch and parse individual article pages if needed

    print(f"\nDownloading article: {article}")
    #if article['pdf'] == '[PDF]': #If there is no PDF available, skip
    link = article['link']
    title = article.get('title').replace('/', '_')
    print(f"\nTrying to download: {title}")
            
    pdf_filename = f"{title}_GS.pdf"

    req = urllib.request.Request(link, headers={'User-Agent': user_agent})
            
    try:

        with urllib.request.urlopen(req) as response:
                    
            pdf_content = response.read()
            os.makedirs('leonel/pdf', exist_ok=True) #Create directory if it doesn't exist
            full_file_path = os.path.join('leonel/pdf', pdf_filename)
            with open(full_file_path, 'wb') as f_pdf:
                       f_pdf.write(pdf_content)
                
            print(f"Download Succesfully. Saved as: {full_file_path}")
            return full_file_path

            
    except urllib.error.HTTPError as e:
        print(f"Error HTTP downloading: {e.code}. The server refused the download.")
    except urllib.error.URLError as e:
        print(f"Error de URL downloading: {e.reason}")
    except Exception as e:
        print(f"Unexpected Error trying to download: {e}")



def fetch_with_retry(req, max_retries=5):
    for attempt in range(max_retries):
        try:
            return urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = random.uniform(30, 60) * (attempt + 1)
                print(f"Erreur 429: Trop de requêtes. Attente de {wait_time:.2f} secondes avant nouvelle tentative ({attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise e
    raise Exception("Echec après plusieurs tentatives (Erreur 429)")

def main():

        file_web = open("leonel/schoolargoogle.html", "w+", encoding='utf-8')
        req = urllib.request.Request(full_url, headers={'User-Agent': user_agent})
        
        # Utilisation de la fonction de retry
        consult = fetch_with_retry(req)
        
        consult_bytes = consult.read()
    
        consult_html = consult_bytes.decode('utf-8')
    
        print("Connection succesful. First 50 characters:")
        print(consult_html[:50])

        file_web.write(str(consult_bytes.decode('utf-8')))
        file_web.close()

        html= open("leonel/schoolargoogle.html", "r+")
        soup = BeautifulSoup(consult_html, 'html.parser')
        class_searched = 'gs_r gs_or gs_scl'
        result = soup.find_all('div', class_=class_searched)
    
        for i, line in enumerate(result):


            title_tag = line.find('h3', class_='gs_rt')
            title = title_tag.text if title_tag else "N/A"
            
            pdf_tag = line.find('span', class_='gs_ctg2')
            pdf = pdf_tag.text if pdf_tag else "N/A"
                
            link_tag = line.find('a')
            link = link_tag['href'] if link_tag else "N/A"

            site_tag = link.split('/')[2].split('.')[0] if link_tag else None
            site = site_tag if site_tag !='www' else link.split('/')[2].split('.')[-2]

            page_tag = line.find('h3', class_='gs_rt')
            page = page_tag.find('a')['href'] if page_tag and page_tag.find('a') else "N/A"

            citations_tag = line.find('div', class_='gs_a')
            citations = citations_tag.text.split('-')[0] if citations_tag else "N/A"
            
            print(f"\n--- Result {i+1} ---")
            print(f"Site: {site}")
            print(f"PDF: {pdf}")
            print(f"Title: {title}")
            print(f"Link: {link}")
            print(f"Authors: {citations}")
            
            log = {
                'site': site,
                'title': title,
                'page': page,
                'link': link,
                'authors': citations,
                'pdf': pdf

            }

            logs_json.append(log)
        #with open('results_google_scholar.json', 'w', encoding='utf-8') as f:
            #json.dump(logs_json, f, ensure_ascii=False, indent=4)
      

        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as f:
                try:
                    file = json.load(f)

                    # Avoid duplicates based on title
                    titles = {item['title'] for item in file}

                except json.JSONDecodeError:
                    file = []
        else: 
            with open(file_name, 'w', encoding='utf-8') as f:
                file = []
                json.dump(file, f, ensure_ascii=False, indent=4)
            titles = set()

        # Append new articles avoiding duplicates
        for article in logs_json:
            if article['title'] not in titles:
                path = get_article(article) # Download the article and get the file path
                
                article['path'] = path
                avoid_duplicates.append(article)


                #I need to add the abstract extraction here
                abstract = get_abstract(article['link']) # Fetch the abstract
                article['abstract'] = abstract
                avoid_duplicates.append(article)
                titles.add(article['title'])
                

        file.extend(avoid_duplicates)

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(file, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
    #article()
