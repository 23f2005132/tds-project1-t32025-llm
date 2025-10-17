# /// script
# This header is for local tools, but requirements.txt is what Hugging Face uses.
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi[standard]",
#   "uvicorn",
#   "requests",
#   "python-dotenv",
#   "google-generativeai"
# ]
# ///

import os
import time
import base64
import json
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks

# --- Configuration & Setup ---
# Load all the secret variables from the .env file
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
API_SECRET = os.getenv("SECRET")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# --- Helper Functions for GitHub ---

def validate_secret(secret: str) -> bool:
    """Checks if the incoming secret matches the one in our .env file."""
    return secret == API_SECRET

def create_github_repo(repo_name: str):
    """Uses the GitHub API to create a new public repository."""
    print(f"Creating GitHub repo named: {repo_name}")
    payload = {
        "name": repo_name,
        "private": False,
        "auto_init": True,  # Creates the repo with a README, essential for the first commit
        "license_template": "mit",
    }
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.post(
        "https://api.github.com/user/repos",
        headers=headers,
        json=payload
    )
    if response.status_code != 201:
        raise Exception(f"Failed to create repo: {response.status_code}, {response.text}")
    print("Repo created successfully.")
    return response.json()

def push_files_to_repo(repo_name: str, files: list[dict]):
    """Pushes a list of files to the repo, handling both creation and updates."""
    print(f"Preparing to push {len(files)} files to {repo_name}...")
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    latest_commit_sha = ""

    for file in files:
        file_name = file.get("name")
        file_content = file.get("content")
        
        encoded_content = base64.b64encode(file_content.encode("utf-8")).decode("utf-8")
        
        url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_name}"
        
        # --- START OF CHANGES ---
        
        # 1. Check if the file already exists to get its SHA
        sha = None
        try:
            get_response = requests.get(url, headers=headers)
            if get_response.status_code == 200:
                # If the file exists, get its SHA
                sha = get_response.json()['sha']
                print(f"File '{file_name}' exists. Preparing to update.")
        except requests.RequestException:
            pass # Proceed without SHA if the check fails

        # 2. Prepare the payload for the PUT request
        payload = {
            "message": f"Add or update file: {file_name}",
            "content": encoded_content
        }
        # If we found a SHA, add it to the payload
        if sha:
            payload['sha'] = sha
            
        # --- END OF CHANGES ---

        print(f"Pushing {file_name}...")
        response = requests.put(url, headers=headers, json=payload)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to push file {file_name}: {response.status_code}, {response.text}")
        
        latest_commit_sha = response.json()["commit"]["sha"]
        print(f"Successfully pushed {file_name}.")
        
    return latest_commit_sha

def get_file_from_repo(repo_name: str, file_path: str):
    print(f"Fetching '{file_path}' from repo '{repo_name}'...")
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Warning: Could not fetch {file_path}. It might not exist. Error: {response.text}")
        return None # Return None if file doesn't exist or on error

    response_data = response.json()
    # Content from GitHub API is Base64 encoded, so we must decode it.
    file_content_encoded = response_data['content']
    file_content_decoded = base64.b64decode(file_content_encoded).decode('utf-8')
    
    print(f"Successfully fetched '{file_path}'.")
    return file_content_decoded

def enable_github_pages(repo_name: str):
    """Enables GitHub Pages for the repository on the main branch."""
    print(f"Enabling GitHub Pages for {repo_name}...")
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "source": {"branch": "main", "path": "/"}
    }
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 201:
        raise Exception(f"Failed to enable GitHub Pages: {response.status_code}, {response.text}")
    print("GitHub Pages enabled. It may take a minute to deploy.")

# --- LLM Code Generation Function ---

def generate_code_with_llm(brief: str, checks: list, attachments: list = None):
    print("Generating code with LLM via Google's SDK...")

    # --- START OF FIX ---
    # We prepare the parts of the prompt separately
    checks_formatted = "\n- ".join(checks)
    
    attachments_formatted = ""
    if attachments:
        attachments_formatted += "\n**Attachments:**"
        for att in attachments:
            header, encoded = att['url'].split(',', 1)
            content_preview = base64.b64decode(encoded).decode('utf-8', errors='ignore')[:200]
            attachments_formatted += f"\n- `{att['name']}`: ```\n{content_preview}...\n```"

    # Use a template string with .format() to avoid the f-string syntax error
    prompt_template = """
You are an expert web developer. Your task is to generate the code for a web app based on the following brief.
You MUST provide your response as a single, valid JSON object with a key "files", which is an array of objects.
Each object must have "name" (e.g., "index.html") and "content" keys.
Include a professional README.md file.

**Brief:**
{brief}

**The app must pass these checks:**
- {checks}
{attachments}
"""
    prompt = prompt_template.format(brief=brief, checks=checks_formatted, attachments=attachments_formatted)
    # --- END OF FIX ---

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        response_content = response.text
        print("LLM code generation successful.")
        return json.loads(response_content)["files"]
    except Exception as e:
        print(f"Error during LLM generation with Google's SDK: {e}")
        raise

# --- Evaluation Notification Function ---

def notify_evaluator(evaluation_url: str, payload: dict):
    """Notifies the evaluation server with deployment details, retrying on failure."""
    max_retries = 4
    delay = 1  # 1, 2, 4, 8 seconds
    for attempt in range(max_retries):
        try:
            print(f"Notifying evaluator... (Attempt {attempt + 1})")
            response = requests.post(evaluation_url, json=payload, timeout=15)
            if response.status_code == 200:
                print("Successfully notified evaluator.")
                return
            else:
                print(f"Evaluator returned status {response.status_code}: {response.text}")
        except requests.RequestException as e:
            print(f"Failed to connect to evaluator: {e}")
        
        time.sleep(delay)
        delay *= 2
    print("Could not notify evaluator after all retries.")

# --- Main Business Logic ---

def process_round1_task(data: dict):
    """The main workflow for a round 1 request."""
    print("--- Starting Round 1 Process ---")
    try:
        # Step 1: Generate code using the LLM
        files_to_commit = generate_code_with_llm(
            brief=data['brief'],
            checks=data['checks'],
            attachments=data.get('attachments')
        )
        
        # Step 2: Create a unique GitHub repository
        repo_name = data["task"]
        repo_info = create_github_repo(repo_name)
        repo_url = repo_info["html_url"]
        
        # Step 3: Push the generated files to the new repository
        # This returns the SHA of the final commit
        commit_sha = push_files_to_repo(repo_name, files_to_commit)
        
        # Step 4: Enable GitHub Pages for the repository
        enable_github_pages(repo_name)
        # Give pages a moment to initialize before forming the URL
        time.sleep(5)
        pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
        
        # Step 5: Prepare payload and notify the evaluation server
        evaluation_payload = {
            "email": data["email"],
            "task": data["task"],
            "round": data["round"],
            "nonce": data["nonce"],
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "pages_url": pages_url,
        }
        notify_evaluator(data["evaluation_url"], evaluation_payload)
        
        print("--- Round 1 Process Completed Successfully ---")
        
    except Exception as e:
        print(f"!!! An error occurred during Round 1 process: {e} !!!")

def process_round2_task(data: dict):
    print("--- Starting Round 2 Process ---")
    try:
        repo_name = data["task"]
        brief = data["brief"]

        # Step 1: Fetch the existing code from the repo.
        # For now, we assume the main file is index.html.
        existing_html = get_file_from_repo(repo_name, "index.html")
        if not existing_html:
            raise Exception("Could not fetch existing index.html from the repo.")

        # Step 2: Create a new, more detailed prompt for the LLM.
        # This prompt includes the new instructions AND the old code.
        revision_prompt_template = """
        You are an expert web developer. Your task is to revise an existing HTML file based on a new brief.
        You MUST provide your response as a single, valid JSON object with a key "files", which is an array of objects.
        Each object must have "name" and "content" keys.
        The main file to modify is "index.html". You must also update the "README.md" to reflect the new changes.

        **New Brief / Revision Request:**
        {brief}

        **Existing index.html Code:**
        ```html
        {existing_html}
        ```
        """
        revision_prompt = revision_prompt_template.format(brief=brief, existing_html=existing_html)
        
        # Step 3: Call the LLM with the new revision prompt.
        updated_files = generate_code_with_llm(
            brief=revision_prompt,
            checks=data.get("checks", [])
        )

        # Step 4: Push the modified files back to the same repo.
        commit_sha = push_files_to_repo(repo_name, updated_files)
        
        # Step 5: Notify the evaluator with the new details for round 2.
        pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"
        repo_url = f"https://github.com/{GITHUB_USERNAME}/{repo_name}"
        evaluation_payload = {
            "email": data["email"], "task": data["task"], "round": data["round"],
            "nonce": data["nonce"], "repo_url": repo_url, "commit_sha": commit_sha,
            "pages_url": pages_url,
        }
        notify_evaluator(data["evaluation_url"], evaluation_payload)

        print("--- Round 2 Process Completed Successfully ---")

    except Exception as e:
        print(f"!!! An error occurred during Round 2 process: {e} !!!")

# --- API Endpoint ---

@app.post("/handle_task/")
def handle_task(data: dict, background_tasks: BackgroundTasks):
    """Receives task, validates secret, and starts processing in the background."""
    if not validate_secret(data.get("secret")):
        return {"status": "error", "message": "Invalid secret"}
    
    # Respond IMMEDIATELY with a 200 OK, as required.
    # The actual work will run in the background.
    print(f"Task received for round {data.get('round')}. Starting in background.")
    if data.get("round") == 1:
        background_tasks.add_task(process_round1_task, data)
        return {"status": "success", "message": "Round 1 task received and is being processed."}
    elif data.get("round") == 2:
        background_tasks.add_task(process_round2_task, data)
        return {"status": "success", "message": "Round 2 task received and is being processed."}
    else:
        return {"status": "error", "message": "Invalid round number"}

@app.get("/")
def read_root():
    return {"message": "LLM Code Deployment Agent is running."}

# This part allows you to run the app locally with `python main.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

