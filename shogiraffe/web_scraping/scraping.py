import requests
from bs4 import BeautifulSoup

# Open the database
file = open("database.txt", mode="wt", encoding="utf-8")

# We look at all the boards possible on the website shogimaze
for i in range(1, 10):
    url = "http://shogimaze.free.fr/usf.php?tesu=10&rank=" + str(i)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    line = soup.find("textarea").contents[0]
    file.write(line)

# Close the database
file.close()


# Clean the database
file = open("database.txt", encoding="utf-8")
data = []
for ln in file:
    if ln.startswith("^"):
        data.append(ln[3:])
file.close()

# Write the final boards after data parsing
file = open("database.txt", mode="wt", encoding="utf-8")
file.writelines(data)
