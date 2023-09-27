import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
import pyterrier as pt

if not pt.started():
    pt.init()

## Open the inverted index
inverted_index_path = "./data/inverted_index"
indexref = pt.IndexRef.of(inverted_index_path)
index = pt.IndexFactory.of(indexref)

for kv in index.getLexicon():
 print((kv.getKey())+"\t"+ kv.getValue().toString())

def main():
    keywords_exp = {"help()" : "print the available commands", 
                "exit()": "to exit the program", 
                "clear()" : "to clear the shell",
                }

    print("Welcome to ChatGPT10")
    end = False
    while not(end):
        query = input('>> Enter your query : ')
        if query == "exit()":
            end = True
        elif query == "clear()":
            if os.name == 'posix':  # For Unix/Linux/Mac systems
                os.system('clear')
            elif os.name == 'nt':   # For windows systems
                os.system('cls')
        elif query == "help()":
            for keyword in keywords_exp:
                print("   - " + keyword + " --> " + keywords_exp[keyword])
# main()