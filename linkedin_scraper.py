import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# --- CONFIG ---
LINKEDIN_EMAIL = 'khyateeswar.hyprbots@gmail.com'
LINKEDIN_PASSWORD = '25@9*w6K!q5Bxm#'
PROFILE_URL = 'https://www.linkedin.com/in/khyateeswar-naidu-282565188/'
OUTPUT_FILE = 'profile_data.json'

# --- SETUP SELENIUM ---
options = Options()
options.add_argument("--start-maximized")
# options.add_argument('--headless')  # Uncomment to run headless
driver = webdriver.Chrome(options=options)

def login_to_linkedin(email, password):
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    driver.find_element(By.ID, "username").send_keys(email)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(3)

def scrape_profile(profile_url):
    driver.get(profile_url)
    time.sleep(5)

    def safe_find(selector, by=By.CSS_SELECTOR):
        try:
            return driver.find_element(by, selector).text.strip()
        except:
            return "Not Found"

    data = {
        "Name": safe_find("h1.text-heading-xlarge"),
        "Headline": safe_find("div.text-body-medium.break-words"),
        "Location": safe_find("span.text-body-small.inline.t-black--light.break-words"),
        "About": safe_find("//section[contains(@class, 'pv-about-section')]//span[1]", by=By.XPATH)
    }

    return data

def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Data exported to {filename}")

# --- RUN ---
login_to_linkedin(LINKEDIN_EMAIL, LINKEDIN_PASSWORD)
profile_data = scrape_profile(PROFILE_URL)

print("\n📄 Scraped Data:")
for key, value in profile_data.items():
    print(f"{key}: {value}")

save_to_json(profile_data, OUTPUT_FILE)
driver.quit()
