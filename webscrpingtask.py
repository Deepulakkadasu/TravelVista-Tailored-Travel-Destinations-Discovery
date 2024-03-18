import requests

# URL of the webpage
url = "https://www.tripadvisor.in/Attractions-g7058854-Activities-oa0-Telangana.html"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Print the HTML content of the response
    print(response.text)
else:
    print("Failed to retrieve the webpage")
