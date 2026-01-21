import streamlit as st
import ast
import numpy as np
import faiss
import pickle
import json
import graphviz
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import anthropic

# === Config ===
NEO4J_URI = "bolt://localhost:7690"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password here"
CLAUDE_API_KEY = "API Key"
INDEX_PATH = "C:/Users/theya/Documents/GitHub/L675_25/graph_rag_pipeline/entity_index.faiss"
NAMES_PATH = "C:/Users/theya/Documents/GitHub/L675_25/graph_rag_pipeline/entity_names.pkl"

# === Init models ===
embedder = SentenceTransformer('all-roberta-large-v1')
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
index = faiss.read_index(INDEX_PATH)
with open(NAMES_PATH, "rb") as f:
    entity_names_faiss = pickle.load(f)

# === Functions ===
def extract_entities_claude(question):
    prompt = f"""
You are a helpful assistant. Extract only the most relevant named entities from this question. 
Only return a valid Python list of strings, like: ["entity1", "entity2"]

Question: "{question}"
"""
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    raw_output = response.content[0].text.strip()
    try:
        return ast.literal_eval(raw_output)
    except Exception:
        return []

def expand_entity(user_entity, top_k=6, threshold=0.6):
    query_emb = embedder.encode(user_entity, convert_to_numpy=True)
    query_emb = np.expand_dims(query_emb, axis=0)
    D, I = index.search(query_emb, top_k)
    matches = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1:
            sim_score = 1 / (1 + dist)
            if sim_score > threshold:
                matches.append((entity_names_faiss[idx], sim_score))
    return [name for name, _ in matches]

def get_filtered_triples(question, entities, similarity_threshold=0.3, top_k=10):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    context_triples = set()
    with driver.session() as session:
        for entity in entities:
            result = session.run("""
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($name) OR toLower(o.name) CONTAINS toLower($name)
                RETURN s.name AS subject, TYPE(r) AS relation, o.name AS object
            """, name=entity)
            for record in result:
                triple_str = f"{record['subject']} {record['relation'].replace('_',' ').lower()} {record['object']}"
                context_triples.add(triple_str)
            if not context_triples:
                for exp in expand_entity(entity):
                    result2 = session.run("""
                        MATCH (s:Entity)-[r]->(o:Entity)
                        WHERE toLower(s.name) CONTAINS toLower($name) OR toLower(o.name) CONTAINS toLower($name)
                        RETURN s.name AS subject, TYPE(r) AS relation, o.name AS object
                    """, name=exp)
                    for record2 in result2:
                        triple_str = f"{record2['subject']} {record2['relation'].replace('_',' ').lower()} {record2['object']}"
                        context_triples.add(triple_str)
    driver.close()
    if not context_triples:
        return []
    context_triples = list(context_triples)
    triple_embeddings = embedder.encode(context_triples, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, triple_embeddings)[0]
    ranked = [(sim, triple) for sim, triple in zip(similarities, context_triples) if sim >= similarity_threshold]
    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)
    return [t for _, t in ranked[:top_k]]

def generate_answer_from_triples_claude(question, triples):
    context = "\n".join([f"- {t}" for t in triples])
    prompt = f"""
You are a biomedical assistant. Only use the information in the following triples to answer the question. Do not use any outside knowledge or make assumptions. If the answer is not present in the triples, respond with: "Not mentioned in the text."

Triples:
{context}

Question: {question}

Give your answer in simple sentence or sentences. No bullet points. No quotes. No newlines. No extra explanation, just stick to the question.
"""
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


from streamlit.components.v1 import html
# === Streamlit UI ===
...

# === Streamlit UI ===
# st.set_page_config(page_title="Biomedical QA", layout="wide", initial_sidebar_state="auto")
# === Streamlit UI ===
# === Streamlit UI ===

# === Functions ===
# [triples retrieval and entity extraction functions assumed available here]
# === UI & Styling Improvements ===
# === UI & Styling Improvements ===
# === UI & Styling Improvements ===
# === UI & Styling Improvements ===
st.markdown("""
<style>
    /* Container styling */
    .block-container {
        padding-top: 1rem;
        max-width: 100% !important;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Text and input styling */
    .css-1v0mbdj p, .stTextInput > label, .stButton > button, .stTable {
        font-size: 18px !important;
    }
    .stTextInput > div > input {
        font-size: 16px !important;
        padding: 0.6rem;
    }
    .css-1aumxhk, .css-1v0mbdj {
        color: #ddd !important;
    }
    
    /* Table styling for full width */
    .element-container:has(.dataframe) {
        width: 100% !important;
    }
    .stDataFrame {
        width: 100% !important;
    }
    .stDataFrame > div {
        width: 100% !important;
    }
    .stDataFrame .css-1v0mbdj {
        font-size: 16px;
    }
    .stDataFrame .css-1v0mbdj td {
        padding: 12px 20px;
    }
    
    /* Layout adjustments for expandable section */
    .streamlit-expanderContent {
        width: 100%;
    }
    
    /* Column adjustments */
    [data-testid="column"]:first-child {
        width: 40% !important;
    }
    [data-testid="column"]:nth-child(2) {
        width: 60% !important;
    }
</style>
""", unsafe_allow_html=True)

# === Enhanced Triple Formatting for Table & Graph ===
def parse_triples(triples_raw):
    parsed = []
    for t in triples_raw:
        try:
            subj, rel, obj = t.split(" -> ", 2)
            parsed.append((subj.strip(), rel.strip(), obj.strip()))
        except:
            continue
    return parsed

# === Updated Triple Retrieval Function ===
def get_filtered_triples(question, entities, similarity_threshold=0.3, top_k=10):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    context_triples = set()
    with driver.session() as session:
        for entity in entities:
            result = session.run("""
                MATCH (s:Entity)-[r]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($name) OR toLower(o.name) CONTAINS toLower($name)
                RETURN s.name AS subject, TYPE(r) AS relation, o.name AS object
            """, name=entity)
            for record in result:
                triple_str = f"{record['subject']} -> {record['relation'].replace('_',' ').lower()} -> {record['object']}"
                context_triples.add(triple_str)
            if not context_triples:
                for exp in expand_entity(entity):
                    result2 = session.run("""
                        MATCH (s:Entity)-[r]->(o:Entity)
                        WHERE toLower(s.name) CONTAINS toLower($name) OR toLower(o.name) CONTAINS toLower($name)
                        RETURN s.name AS subject, TYPE(r) AS relation, o.name AS object
                    """, name=exp)
                    for record2 in result2:
                        triple_str = f"{record2['subject']} -> {record2['relation'].replace('_',' ').lower()} -> {record2['object']}"
                        context_triples.add(triple_str)
    driver.close()
    if not context_triples:
        return []
    context_triples = list(context_triples)
    triple_embeddings = embedder.encode(context_triples, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, triple_embeddings)[0]
    ranked = [(sim, triple) for sim, triple in zip(similarities, context_triples) if sim >= similarity_threshold]
    ranked = sorted(ranked, key=lambda x: x[0], reverse=True)
    print("123", [t for _, t in ranked[:top_k]])
    return [t for _, t in ranked[:top_k]]

def render_d3_graph(triples_raw):
    triples = parse_triples(triples_raw)
    nodes = set()
    edges = []
    for subj, rel, obj in triples:
        nodes.add(subj)
        nodes.add(obj)
        edges.append({"source": subj, "target": obj, "label": rel})

    node_list = [{"id": n} for n in nodes]
    edge_list = [{"source": e["source"], "target": e["target"], "label": e["label"]} for e in edges]

    html_code = f"""
    <div id='graph'></div>
    <script src='https://d3js.org/d3.v7.min.js'></script>
    <script>
    const nodes = {json.dumps(node_list)};
    const links = {json.dumps(edge_list)};

    // Reduced width for more compact graph
    const width = 600;
    const height = 500;

    // Create SVG container with zoom behavior
    const svg = d3.select("#graph").append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", "0 0 " + width + " " + height)
        .attr("style", "max-width: 100%; height: auto;");
        
    // Add zoom functionality to the SVG
    const g = svg.append("g");
    
    svg.call(d3.zoom()
        .scaleExtent([0.1, 4])
        .on("zoom", function(event) {{
            g.attr("transform", event.transform);
        }}));

    // Create simulation with more controlled forces for a more compact layout
    const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id(d => d.id).distance(100))  // Reduced distance
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("x", d3.forceX(width / 2).strength(0.15))
        .force("y", d3.forceY(height / 2).strength(0.15))
        .force("collide", d3.forceCollide().radius(40));  // Reduced collision radius

    // Create links
    const link = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke", "#888")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", 1.5);
        
    // Create link labels with smaller font
    const linkLabels = g.append("g")
        .selectAll(".link-label")
        .data(links)
        .join("text")
        .attr("class", "link-label")
        .attr("font-size", 10)  // Reduced font size
        .attr("fill", "#aaa")
        .attr("text-anchor", "middle")
        .text(d => d.label);

    // Create nodes
    const node = g.append("g")
        .selectAll(".node")
        .data(nodes)
        .join("g")
        .attr("class", "node")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));
            
    // Add smaller circles to nodes
    node.append("circle")
        .attr("r", 8)  // Reduced radius
        .attr("fill", "#49c3b1");
        
    // Add text labels to nodes with smaller font
    node.append("text")
        .attr("dx", 12)  // Reduced offset
        .attr("dy", 3)
        .attr("font-size", 11)  // Reduced font size
        .attr("fill", "#ddd")
        .text(d => d.id);
    
    // Define drag functions
    function dragstarted(event, d) {{
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }}
    
    function dragged(event, d) {{
        d.fx = event.x;
        d.fy = event.y;
    }}
    
    function dragended(event, d) {{
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }}

    // Update positions on simulation tick
    simulation.on("tick", () => {{
        // Constrain nodes to stay within bounds
        nodes.forEach(d => {{
            d.x = Math.max(40, Math.min(width - 40, d.x));
            d.y = Math.max(40, Math.min(height - 40, d.y));
        }});
        
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);
            
        linkLabels
            .attr("x", d => (d.source.x + d.target.x) / 2)
            .attr("y", d => (d.source.y + d.target.y) / 2);
            
        node
            .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
    }});
    
    // Run simulation for a few iterations before displaying
    simulation.tick(20);
    </script>
    """
    html(html_code, height=550)  # Reduced height
# === Step 2 Table Update ===
def show_triples_table(triples_raw):
    parsed = parse_triples(triples_raw)
    df = [{"Subject": s, "Predicate": p, "Object": o} for s, p, o in parsed]
    st.dataframe(df, use_container_width=True)
    return parsed
st.title("🔬 Improving LLM Reasoning with Knowledge Graphs")

question = st.text_input("Enter your biomedical question:")

if st.button("Generate Answer") and question:
    with st.spinner("Processing..."):
        entities = extract_entities_claude(question)
        triples = get_filtered_triples(question, entities)
        answer = generate_answer_from_triples_claude(question, triples)

        html(f"""
        <div style='background-color:#d4edda;padding:10px;border-radius:5px;border:1px solid #c3e6cb;margin-top:10px'>
            <strong>Answer:</strong> {answer}
        </div>
        """, height=60)

# Apply the updated styles

    with st.expander("🔍 How was this answer generated?"):
        # Updated column ratio to give more space to the table
        col1, col2 = st.columns([3, 2])  # Changed from [1, 2] to [3, 2]

        with col1:
            st.markdown("**Step 1: Extracted Entities**")
            st.write(entities)

            st.markdown("**Step 2: Top Distinct Triples Used**")
            # Display the table with extra width
            parsed = show_triples_table(triples)

        with col2:
            st.markdown("**Triple Visualization (Interactive)**")
            # More compact graph rendering
            render_d3_graph(triples)

        if not triples:
            st.warning("No relevant triples were found in Neo4j.")