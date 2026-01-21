from neo4j import GraphDatabase
import ast
import re

# === CONFIG ===
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password here"
INPUT_FILE = "C:/Users/theya/Downloads/pmid_triples.txt"
FAILED_OUTPUT_FILE = "C:/Users/theya/Downloads/failed_triples.txt"

# === UTILS ===
def normalize(text):
    return text.replace("'", "\\'")

def predicate_to_relation(pred):
    rel = pred.strip().upper().replace(" ", "_")
    rel = re.sub(r"[^A-Z0-9_]", "", rel)
    if not rel or not rel[0].isalpha():
        rel = "REL_" + rel
    return rel

def parse_triple_line(line):
    return ast.literal_eval(line.strip())

# === INSERT FUNCTION with Error Handling ===
def upload_triples(driver, triples, keep_pubmed=True):
    failed = []

    with driver.session() as session:
        for pubmedid, subj, pred, obj in triples:
            try:
                subj = normalize(subj)
                obj = normalize(obj)
                rel = predicate_to_relation(pred)
                pubmedid = pubmedid.strip()

                if keep_pubmed:
                    query = f"""
                    MERGE (s:Entity {{name: '{subj}'}})
                    MERGE (o:Entity {{name: '{obj}'}})
                    MERGE (s)-[r:{rel}]->(o)
                    SET r.source = '{pubmedid}'
                    """
                else:
                    query = f"""
                    MERGE (s:Entity {{name: '{subj}'}})
                    MERGE (o:Entity {{name: '{obj}'}})
                    MERGE (s)-[:{rel}]->(o)
                    """

                session.run(query)

            except Exception as e:
                failed.append((pubmedid, subj, pred, obj, str(e)))

    # Save failures
    if failed:
        with open(FAILED_OUTPUT_FILE, "w", encoding="utf-8") as f:
            for row in failed:
                f.write(f"{row}\n")
        print(f"❌ {len(failed)} triples failed. Logged to {FAILED_OUTPUT_FILE}")

# === MAIN ===
if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        triples = [parse_triple_line(line) for line in file if line.strip()]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    print(f"Uploading {len(triples)} triples to Neo4j...")
    upload_triples(driver, triples, keep_pubmed=True)
    print("✅ Done.")

    driver.close()
