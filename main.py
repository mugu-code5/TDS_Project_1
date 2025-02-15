# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "python-dateutil",
#     "httpx",
#     "fastapi[all]",
#     "numpy",
#     "uvicorn",
#     "Pillow",
#     "speechrecognition",
#     "requests",
#     "markdown",
#      "aiofiles"
# ]
# ///

from fastapi import FastAPI, HTTPException, Query
import httpx
import base64
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import json
from dateutil.parser import parse
import subprocess
import glob
import numpy as np
import re
import sqlite3
import traceback
import aiofiles
import requests
import os
import markdown
from PIL import Image
import speech_recognition as sr


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN") 

@app.get("/read")
async def read_file(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
headers = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json",
}


Script_to_run={
        "type": "function", # ---> A1
        "function": {
            "name": "Script_to_run",
            "description": "Install a package and run a script from a URL with provided arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string","description": "The URL of the script to run."},
                    "email": {"type": "array","items": {"type": "string"}, "description": "List of arguments to pass to the script"}
                },
                "required": ["url", "email"]
            }
        }
    }
Format_prettier={
        "type": "function", # ---> A2
        "function": {
            "name": "Format_prettier",
            "description": "Format a file using Prettier with a specific prettier version",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_path": {"type": "string", "description": "The path to the file to be formatted (e.g., ./data/format.md)"}
                },
                "required": ["input_file_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
count_day={
        "type": "function", # ---> A3
        "function": { 
            "name": "count_day",
            "description": "Count the number of specific weekdays from a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_path": {"type": "string","description": "The location of the input file containing dates"},
                    "output_file_path": {"type": "string","description": "The location where the output count will be written"},
                    "day": {"type": "string", "description": "The day of the week to count (e.g., 'Monday', 'Tuesday', etc.)"}
                },
                "required": ["input_file_path", "output_file_path", "day"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
sort_contacts_json={
        "type": "function", # ---> A4
        "function": {
            "name": "sort_contacts_json",
            "description": "Sort contacts by specified fields and write to an output file",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_location": { "type": "string", "description": "The path to the input JSON file containing contacts (e.g., ./data/contacts.json)"},
                    "output_file_location": { "type": "string", "description": "The path to the output JSON file where sorted contacts will be written (e.g., ./data/sorted_contacts.json)"},
                    "first_sort": {"type": "string","description": "The first field to sort by (e.g., 'last_name')"},
                    "second_sort": {"type": "string","description": "The second field to sort by (e.g., 'first_name')"}
                },
                "required": ["input_file_location", "output_file_location", "first_sort", "second_sort"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
most_recent={
        "type": "function", # ---> A5
        "function": {
            "name": "most_recent",
            "description": "Write the specified line from the most recent .log files to an output file",
            "parameters": {
                "type": "object",
                "properties": {
                    "log": {"type": "integer", "description": "The number of most recent log files to read (e.g., 10)"},
                    "line_no": {"type": "integer", "description": "The line number to extract from each log file (e.g., 1 for the first line)"},
                    "input_file_location": {"type": "string", "description": "The path to the directory containing the log files (e.g., /data/logs)"},
                    "output_file_location": {"type": "string", "description": "The path to the output file (e.g., /data/logs-recent.txt)"}
                },
                "required": ["log", "line_no", "input_file_location", "output_file_location"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Markdown_to_index={
        "type": "function", #  ---> A6
        "function": {
            "name": "Markdown_to_index",
            "description": "Create an index file for Markdown files based on specified tag and occurrence",
            "parameters": {
                "type": "object",
                "properties": {
                    "given_file_type": {"type": "string","description": "The file extension/type to search for (e.g., .md)"},
                    "input_file_location": {"type": "string","description": "The path to the directory containing the files (e.g., /data/docs)"},
                    "output_file_location": {"type": "string","description": "The path to the output JSON file (e.g., /data/docs/index.json)"},
                    "occ": {"type": "integer","description": "Which occurrence of the tag to look for (e.g., 1 for first occurrence)"},
                    "tags": {"type": "string","description": "The Markdown header tag to extract (e.g., #, ##, ###)"}
                },
                "required": ["given_file_type", "input_file_location", "output_file_location", "occ", "tags"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Extract_email={
        "type": "function",  #  ---> A7
        "function": {
            "name": "Extract_email",
            "description": "Extract the sender's email address from an input file and write just the email address to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file_location": {"type": "string","description": "The location of the input file."},
                    "output_file_location": {"type": "string","description": "The location of the output file."}
                },
                "required": ["input_file_location", "output_file_location"]
            }
        }
    }
Card_number={
        "type": "function",  # ---> A8
        "function": {
            "name": "Card_number",
            "description": "Extract a credit card number from an image and save it to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string","description": "The path to the image file (e.g., /data/images/card.png)"},
                    "output_path": {"type": "string","description": "The path to the output file (e.g., /data/output/card_number.txt)"}
                },
                "required": ["image_path", "output_path"],
                "additionalProperties": False
            },"strict": True
        }
    }
Most_similare_comments={
        "type": "function", # ---> A9
        "function": {
            "name": "Most_similare_comments",
            "description": "Compare comments from a file and save the most similar comments to an output file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_file": {"type": "string","description": "The path to the file containing the comments (e.g., /data/comments.txt)"},
                    "output_file": {"type": "string","description": "The path to the output file where the most similar comments will be saved (e.g., /data/most_similar_comments.txt)"}
                },
                "required": ["input_file", "output_file"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Concert_ticket={
    "type": "function",  # ---> A10
    "function": {
        "name": "Concert_ticket",
        "description": "Calculate the total sales for 'Gold' ticket types and write the result to a file",
        "parameters": 
        {    
            "type": "object",
            "properties": {
                "database_file_location": {"type": "string","description": "The location of the SQLite database file (e.g., /data/ticket-sales.db)"},
                "output_file_location": {"type": "string","description": "The location where the total sales for 'Gold' ticket types will be written (e.g., /data/ticket-sales-gold.txt)"}
            },
            "required": ["database_file_location", "output_file_location"],
            "additionalProperties": False
        },
        "strict": True
    }
}
A_function_calls = [Script_to_run,Concert_ticket,Most_similare_comments,Card_number,Extract_email,Markdown_to_index,most_recent,sort_contacts_json,count_day,Format_prettier]

Markdown_to_html={
        "type": "function",
        "function": {
            "name": "Markdown_to_html",
            "description": "Convert a Markdown file to HTML and save it to a specified output path",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_location": {"type": "string", "description": "The path to the input Markdown file"},
                    "output_location": {"type": "string", "description": "The path to save the converted HTML file"}
                },
                "required": ["input_location", "output_location"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Clone={
        "type": "function",
        "function": {
            "name": "Clone",
            "description": "Clone a Git repository, create a new file, commit the changes, and push them to the repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "The URL of the Git repository to clone"},
                    "commit_message": {"type": "string", "description": "The commit message for the changes"},
                    "file_name": {"type": "string", "description": "The name of the new file to create in the repository"},
                    "file_content": {"type": "string", "description": "The content to write into the new file"}
                },
                "required": ["repo_url", "commit_message", "file_name", "file_content"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Fetching_api_data={
        "type": "function",
        "function": {
            "name": "Fetching_api_data",
            "description": "Fetch data from an API and save it to a specified path",
            "parameters": {
                "type": "object",
                "properties": {
                    "api_url": {"type": "string", "description": "The URL of the API to fetch data from"},
                    "save_path": {"type": "string", "description": "The path to save the fetched data (e.g., /data/data.json)"}
                },
                "required": ["api_url", "save_path"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Sql_querys={
        "type": "function",
        "function": {
            "name": "Sql_querys",
            "description": "Run an SQL query on a specified SQLite database",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_path": {"type": "string", "description": "The path to the SQLite database file"},
                    "query": {"type": "string", "description": "The SQL query to be executed"}
                },
                "required": ["database_path", "query"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Image_formate={
        "type": "function",
        "function": {
            "name": "Image_formate",
            "description": "Resize an image and save it to a specified output path",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "The path to the input image file"},
                    "output_path": {"type": "string", "description": "The path to save the resized image (e.g., /data/resized_image.jpg)"},
                    "width": {"type": "integer", "description": "The width of the resized image in pixels"},
                    "height": {"type": "integer", "description": "The height of the resized image in pixels"}
                },
                "required": ["image_path", "output_path", "width", "height"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
audio_to_transcribe={
        "type": "function",
        "function": {
            "name": "audio_to_transcribe",
            "description": "Transcribe audio from an MP3 file to text and save it to a specified output path",
            "parameters": {
                "type": "object",
                "properties": {
                    "inputfile": {"type": "string", "description": "The path or URL of the input MP3 file"},
                    "outputfile": {"type": "string", "description": "The path to save the transcribed text (e.g., /data/transcription.txt)"}
                },
                "required": ["inputfile", "outputfile"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
Delete={
        "type": "function",
        "function": {
            "name": "Delete",
            "description": "Data is never deleted anywhere on the file system, even if the task description asks for it",
        }
    }
    
B_function_calls=[Markdown_to_html,Clone,Fetching_api_data,Sql_querys,Image_formate,audio_to_transcribe,Delete]


async def Script_to_run(url, email):
    email = " ".join(email) if isinstance(email, list) else email
    executed = ["uv", "run", url, email,"--reload","./data"]
    subprocess.run(executed)
    return "Command executed successfully data as be downloaded"


def Format_prettier(input_file_path: str):
    print(f"Input file path: {input_file_path}")
    if not os.path.isfile(input_file_path):
        print(f"Incorrect file path: {input_file_path}")
        return "Incorrect file path"
    
    npx = "npx"
    try:
        process = subprocess.run([npx, f"prettier@3.4.2", "--write", input_file_path], check=True, capture_output=True, text=True, shell=True)
        print(f"Successfully formatted the file {input_file_path} using Prettier@3.4.2")
        print(process.stdout)
        
    except subprocess.CalledProcessError as error:
        print(f"Error during formatting the file : {error}")
        print(error.output)
        
    except FileNotFoundError as error:
        print(f"File not found: {error}")


async def count_day(input_file_path: str, output_file_path: str, weekday: str):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_index = days.index(weekday)
    input_file_path = input_file_path
    output_file_path = output_file_path
    count = 0
    try:
        with open(input_file_path, "r") as file:
            for line in file:
                date_str = line.strip()
                if not date_str:
                    continue 
                try:
                    parsed_date = parse(date_str)
                    if parsed_date.weekday() == day_index:
                        count += 1
                except ValueError:
                    print(f"Skipping invalid date format: {date_str}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    try:
        with open(output_file_path, "w") as file:
            file.write(str(count))
        print(f"Successfully wrote the counts to this {output_file_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

    return f'Number of {weekday}s: {count}'


async def sort_contacts_json(input_file_location,output_file_location,first_sort, second_sort):
    input_file_location= os.path.abspath(input_file_location)
    output_file_location = os.path.abspath(output_file_location)
    with open(os.path.abspath(input_file_location), 'r') as file:
        contacts = json.load(file)
    sorted_contacts = sorted(contacts, key=lambda x: (x[first_sort], x[second_sort]))
    with open(os.path.abspath(output_file_location), 'w') as file:
        json.dump(sorted_contacts, file, indent=4)
    return(f"Contacts sorted and written to .{os.path.abspath(output_file_location)}")


def most_recent(log, line_no, input_file_location, output_file_location):
    input_file_location = os.path.abspath(input_file_location)
    output_file_location = os.path.abspath(output_file_location)
    log_files = glob.glob(os.path.join(input_file_location, '*.log'))
    log_files.sort(key=os.path.getmtime, reverse=True)
    recent_logs = log_files[:log]
    
    with open(output_file_location, 'w', encoding='utf-8') as output_file:
        for log_file in recent_logs:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if line_no - 1 < len(lines):
                    output_file.write(lines[line_no - 1] + '\n')
    return f"The most_recent are print to the ouput file {output_file_location}"


def Markdown_to_index(given_file_type, input_file_location, output_file_location, occ, tags):
    index = {}
    tag_prefix = f"{tags} "
    input_file_location= os.path.abspath(input_file_location)
    output_file_location = os.path.abspath(output_file_location)
    for root, _, files in os.walk(input_file_location):
        for file in files:
            if file.endswith(given_file_type):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    count = 0
                    for line in f:
                        if line.startswith(tag_prefix):
                            count += 1
                            if count == occ:
                                title = line.strip(tag_prefix).strip()
                                relative_path = os.path.relpath(filepath, input_file_location)
                                index[relative_path] = title
                                break
    
    with open(output_file_location, 'w', encoding='utf-8') as index_file:
        json.dump(index, index_file, indent=4)

    return f"The file as successfully created in this loc {output_file_location}"



async def Extract_email(email_content: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Extract the sender's email address from the following email message:\n\n" + email_content}]
            },
        )
    
    response.raise_for_status()
    received_response = response.json()
    content = received_response.get("choices", [{}])[0].get("message", {}).get("content", "")

    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', content)
    return match.group(0) if match else ""


async def load_comments(file_path):
    """Load comments from file asynchronously."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

async def get_embeddings(comments):
    batch_no = 1000 
    """Fetch the embeddings in batches for more efficiency."""
    embeddings = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(comments), batch_no):
            batch = comments[i : i + batch_no]
            response = await client.post(URL, headers=headers, json={"model": "text-embedding-3-small", "input": batch})
            response.raise_for_status()
            embeddings.extend([item["embedding"] for item in response.json()["data"]])
    return np.array(embeddings)

def find_most_similar_fast(comments, embeddings):
    """Find the most similar comments using vectorized cosine similarity."""
    similarity_matrix = np.dot(embeddings, embeddings.T) / (
        np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(embeddings, axis=1)
    )
    np.fill_diagonal(similarity_matrix, -1)
    i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    return comments[i], comments[j]


async def Card_number(image_path: str, output_path: str):
    try:
        image_path = Path(image_path)  
        output_path = Path(output_path)
        if not image_path.exists():
            return f"Error: Image file '{image_path}' not found."

        # Read and encode the image in Base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "Analyze the following image and provide me with any numbers you can find."},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ],
                    "max_tokens": 50
                }
            )

            response_data = response.json()
            print("API Response:", json.dumps(response_data, indent=4))  # Debugging output

            card_numbers = (
                response_data.get("choices", [{}])[0]
                .get("message", {}).get("content", "")
                .replace(" ", "")
            )
            contents = "".join(response_data["choices"][0]["message"]['content'])
            if not card_numbers:
                return "Failed to extract credit card number."
            
            match = re.search(r'\b\d{4} \d{4} \d{4} \d{4}\b', contents)
            # Return the matched 16-digit number if found, else return None
            card = match.group() if match else None

            with open(output_path, "w") as output_file:
                output_file.write(card)
            return f"Credit card number extracted and saved to {output_path}"
    
    except Exception as e:
        print("An error occurred:", e)
        traceback.print_exc()


async def Most_similare_comments(input_file, output_file):
    """Find and save the most similar comments asynchronously."""
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)
    Comment = await load_comments(input_file)
    if len(Comment) < 2:
        return "There is no two or more comment in the given file."

    embeddings = await get_embeddings(Comment)
    Comment1, Comment2 = find_most_similar_fast(Comment, embeddings)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{Comment1}\n{Comment2}\n")

    return f" Comments saved to the given output file {output_file}"


def Concert_ticket(database_file_location: str, output_file_location: str):

    connection= sqlite3.connect(database_file_location)
    cursor = connection.cursor()
    query = "SELECT SUM(units * price) as total_sales FROM tickets WHERE type = 'Gold'"
    cursor.execute(query)
    output = cursor.fetchone()[0]
    with open(output_file_location, 'w') as file:
        file.write(str(output))
    cursor.close()
    connection.close()
    return output



async def Markdown_to_html(input_location: str, output_location: str):
    if not await is_within_data_directory(input_location) or not await is_within_data_directory(output_location):
        raise HTTPException(status_code=403, detail="Paths must be within /data directory.")
    async with aiofiles.open(input_location, 'r') as file:
        markdown_content = await file.read()
    html_content = markdown.markdown(markdown_content)
    async with aiofiles.open(output_location, 'w') as file:
        await file.write(html_content)
    return {"message": "Markdown converted to HTML successfully"}

def Clone(repo_url: str, commit_message: str, file_name: str, file_content: str):
    try:
        # Get the repo name from the URL
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        
        # Clone the repository
        subprocess.run(['git', 'clone', repo_url])
        
        # Change directory to the cloned repository
        os.chdir(repo_name)
        
        # Create a new file and write content to it
        with open(file_name, 'w') as file:
            file.write(file_content)
        
        # Stage the file for commit
        subprocess.run(['git', 'add', file_name])
        
        # Commit the changes
        subprocess.run(['git', 'commit', '-m', commit_message])
        
        # Push the changes to the remote repository
        subprocess.run(['git', 'push'])
        
        return {"message": f"Changes committed and pushed to {repo_url}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def Fetching_api_data(api_url: str, save_path: str):
    if not await is_within_data_directory(save_path):
        raise HTTPException(status_code=403, detail="Save path must be within /data directory.")
    response = requests.get(api_url)
    data = response.json()
    async with aiofiles.open(save_path, 'w') as file:
        await file.write(json.dumps(data))
    return {"message": "Data fetched and saved successfully"}

async def Sql_querys(database_path: str, query: str):
    if not await is_within_data_directory(database_path):
        raise HTTPException(status_code=403, detail="Database path must be within /data directory.")
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return {"results": results}

async def Image_formate(image_path: str, output_path: str, width: int, height: int):
    if not await is_within_data_directory(image_path) or not await is_within_data_directory(output_path):
        raise HTTPException(status_code=403, detail="Image paths must be within /data directory.")
    image = Image.open(image_path)
    image = image.resize((width, height))
    image.save(output_path)
    return {"message": "Image resized successfully"}

def audio_to_transcribe(inputfile, outputfile):
    try:
        if inputfile.startswith('http://') or inputfile.startswith('https://'):

            response = requests.get(inputfile)
            with open('temp_audio.mp3', 'wb') as f:
                f.write(response.content)
            inputfile = 'temp_audio.mp3'
        

        wav_file = 'temp_audio.wav'
        subprocess.run(['ffmpeg', '-i', inputfile, wav_file])

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)

        transcription = recognizer.recognize_google(audio_data)

        with open(outputfile, 'w', encoding='utf-8') as file:
            file.write(transcription)

        os.remove(wav_file)
        if 'temp_audio.mp3' in locals():
            os.remove('temp_audio.mp3')

        print(f"Transcription successful! Saved to {outputfile}")

    except Exception as e:
        print(f"An error occurred: {e}")


    
async def is_within_data_directory(file_path: str) -> bool:
    data_directory = os.path.abspath("data")
    file_path = os.path.abspath(file_path)
    return os.path.commonpath([data_directory]) == os.path.commonpath([data_directory, file_path])


async def main(task: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": task}],
                "tools": A_function_calls+B_function_calls,
                "tool_choice": "auto",
            },
        )
    
    received_response = response.json()
    
    if "choices" not in received_response or not received_response["choices"]:
        raise HTTPException(status_code=500, detail="Invalid response from OpenAI API (no choices).")
    choice = received_response["choices"][0]["message"]
    if "tool_calls" not in choice or not choice["tool_calls"]:
        raise HTTPException(status_code=500, detail="Invalid response from OpenAI API (no tool_calls).")
    
    return received_response

@app.get("/run")
async def task(task: str = Query(..., description="The task to perform")):
    try:
        result = await main(task)
        tools_call = result["choices"][0]["message"]["tool_calls"][0] if "tool_calls" in result["choices"][0]["message"] else None
        function_name = tools_call["function"]["name"]
        argumentes = tools_call["function"]["arguments"]
        arges = json.loads(argumentes)  

        output = result
        function_call = output["choices"][0]["message"]["tool_calls"][0]["function"]
        args = json.loads(function_call["arguments"])
        
        if tools_call:
            if function_name == "Delete":
                raise HTTPException(status_code=403, detail="File deletion is not allowed.")

            elif function_name == "Markdown_to_html":
                input_file_location= os.path.abspath(arges["input_location"])
                output_file_location = os.path.abspath(arges["output_location"])
                return await Markdown_to_html(input_file_location,output_file_location )
            
            elif function_name == "Clone":
                return Clone( arges["repo_url"], arges["commit_message"], arges["file_name"],arges["file_content"])
            
            elif function_name == "Fetching_api_data":
                file_path = os.path.abspath(arges["save_path"])
                return await Fetching_api_data(arges["api_url"],file_path)
            
            elif function_name == "Sql_querys":
                db_path= os.path.abspath(arges["database_path"])
                querys=arges["query"]
                return await Sql_querys(db_path, querys)
            
            elif function_name == "Image_formate":
                image_path= os.path.abspath(arges["image_path"])
                output_path= os.path.abspath(arges["output_path"])
                given_width=arges["width"]
                given_height=arges["height"]
                return await Image_formate(image_path, output_path, given_width, given_height)
            
            elif function_name == "audio_to_transcribe":
                inputfile= os.path.abspath(arges["inputfile"])
                outputfile=  os.path.abspath(arges["outputfile"])
                return audio_to_transcribe(inputfile,outputfile)
            

        
            elif function_call["name"] == "Script_to_run":                                                 #A1
                print("Script is running ")
                print(args["url"] , args["email"] )
                program_1= await Script_to_run(url=args["url"], email=args["email"] )   
                return program_1
            
            elif function_call["name"] == "Format_prettier":                                             #A2
                Format_prettier(args["input_file_path"])
                return "Given file is formatted successfully using prettier@3.4.2 "
            
            elif function_call["name"] == "count_day":                                                  #A3
                program_3 = await count_day(args["input_file_path"], args["output_file_path"], args["day"])
                return program_3
            
            elif function_call["name"] == "sort_contacts_json":                                          #A4
                program_4 = await sort_contacts_json(args["input_file_location"], args["output_file_location"], args["first_sort"], args["second_sort"])
                return  program_4
            
            elif function_call["name"] == "most_recent":                                                   #A5
                program_5 = most_recent(args["log"], args["line_no"], args["input_file_location"],  args["output_file_location"])
                return  program_5
            
            elif function_call["name"] == "Markdown_to_index":                                              #A6
                program_6 = Markdown_to_index(args["given_file_type"],args["input_file_location"],args["output_file_location"], args["occ"], args["tags"])
                return program_6
            
            elif function_call["name"] == "Extract_email":                                                  #A7
                input_file = args["input_file_location"]
                output_file = args["output_file_location"]
                
                with open(input_file, "r", encoding="utf-8") as f:
                    email_content = f.read()
                
                sender_email = await Extract_email(email_content)
                
                if sender_email:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(sender_email)
                    return {"The sender email is ": sender_email}
                else:
                    return("No email address found in the send request.")
                
            elif function_call["name"] == "Card_number":                                                     # A8
                image_file_path = os.path.abspath(args["image_path"])
                output_file_path = os.path.abspath(args["output_path"])
                result = await Card_number(image_file_path, output_file_path)
                return result 
            
            
            elif function_call["name"] == "Most_similare_comments":                                             #A9
                program_9 = await Most_similare_comments( args["input_file"],args["output_file"])
                return  program_9
            
            elif function_call["name"] == "Concert_ticket":                                                    #A10
                database_file_loc = os.path.abspath(args["database_file_location"])
                output_file_loc = os.path.abspath(args["output_file_location"])
                print(database_file_loc,output_file_loc)
                Concert_ticket(database_file_loc, output_file_loc)
                return "Concert_ticket the file formatted successfully"
            else:
                return {"result": "Function call is not present in the given list."}
        else:
            return {"result": result["choices"][0]["message"]["content"]}

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
