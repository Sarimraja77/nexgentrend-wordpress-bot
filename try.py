import streamlit as st
import streamlit.components.v1 as components
import re
import os
import base64
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()


STORE_DOMAIN = os.getenv("SHOPIFY_STORE_DOMAIN")
STOREFRONT_TOKEN = os.getenv("SHOPIFY_STOREFRONT_PUBLIC_TOKEN")
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def fetch_products():
    """Fetch live products from Shopify Storefront API."""
    url = f"https://{STORE_DOMAIN}/api/2024-10/graphql.json"
    query = """
    {
      products(first: 50) {
        edges {
          node {
            title
            description
            priceRange {
              minVariantPrice {
                amount
                currencyCode
              }
            }
            totalInventory
            rating: metafield(namespace: "custom", key: "rating") {
              value
            }
            featuredImage {
              url
            }
            variants(first: 1) {
              edges {
                node {
                  id
                }
              }
            }
          }
        }
      }
    }
    """
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
    }
    response = requests.post(url, json={"query": query}, headers=headers)
    data = response.json()
    products = {}
    for edge in data["data"]["products"]["edges"]:
        node = edge["node"]
        name = node["title"]
        price = node["priceRange"]["minVariantPrice"]["amount"]
        currency = node["priceRange"]["minVariantPrice"]["currencyCode"]
        # "N/A" is language-neutral so the LLM won't try to translate it
        # and risk producing garbled Arabic diacritics.
        rating = node["rating"]["value"] if node["rating"] else "N/A"
        variant_id = node["variants"]["edges"][0]["node"]["id"] if node["variants"]["edges"] else None
        image_url = node["featuredImage"]["url"] if node.get("featuredImage") else None
        products[name] = {
            "Price": price,
            "Currency": currency,
            "Stock": node["totalInventory"],
            "Rating": rating,
            "description": node["description"],
            "variant_id": variant_id,
            "image_url": image_url,
        }
    return products

@st.cache_data(ttl=300)  # refresh every 5 minutes
def get_products():
    return fetch_products()

def describe_uploaded_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Use Groq's vision model to describe a user-uploaded product photo."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    vision_llm = ChatGroq(model=VISION_MODEL, temperature=0)
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe this product photo in 2-3 concise sentences: what type of "
                    "product it is, its color, material, and any distinguishing features. "
                    "This description will be used to match it against a store catalog."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
            },
        ]
    )
    response = vision_llm.invoke([message])
    return response.content

@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, stock, rating, description and image URL."""
    products = get_products()
    products_lookup = {k.lower(): v for k, v in products.items()}
    p = products_lookup.get(name.lower())
    if not p:
        return f"product not found. Available: {', '.join(products)}"
    return str(p)

@tool
def list_products() -> str:
    """List all available products with their prices, stock, rating and image URL."""
    products = get_products()
    lines = [
        f"{name}: {info['Price']} {info['Currency']} | Stock: {info['Stock']} | "
        f"Rating: {info['Rating']} | Image: {info['image_url']}"
        for name, info in products.items()
    ]
    return "\n".join(lines)

def create_cart(variant_id, quantity=1):
    """Create a Shopify cart with the given variant and return checkout URL."""
    url = f"https://{STORE_DOMAIN}/api/2024-10/graphql.json"
    mutation = """
    mutation($variantId: ID!, $qty: Int!) {
      cartCreate(input: {
        lines: [{ merchandiseId: $variantId, quantity: $qty }]
      }) {
        cart {
          id
          checkoutUrl
        }
        userErrors {
          field
          message
        }
      }
    }
    """
    variables = {"variantId": variant_id, "qty": quantity}
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Storefront-Access-Token": STOREFRONT_TOKEN,
    }
    response = requests.post(url, json={"query": mutation, "variables": variables}, headers=headers)
    return response.json()

@tool
def add_to_cart(product_name: str, quantity: int = 1) -> str:
    """Add a product to the cart by name and return a checkout link. Use the exact product name from the catalog."""
    products = get_products()
    products_lookup = {k.lower(): v for k, v in products.items()}
    p = products_lookup.get(product_name.lower())
    if not p:
        return f"Product not found. Available: {', '.join(products)}"
    if not p.get("variant_id"):
        return "Sorry, this product cannot be added to cart right now."

    result = create_cart(p["variant_id"], quantity)
    cart_data = result.get("data", {}).get("cartCreate", {})
    errors = cart_data.get("userErrors", [])
    if errors:
        return f"Could not add to cart: {errors[0]['message']}"

    checkout_url = cart_data["cart"]["checkoutUrl"]
    numeric_variant_id = p["variant_id"].split("/")[-1]

    # Special marker so the UI layer knows to sync the site's real cart
    marker = f"[[SYNC_CART:{numeric_variant_id}:{quantity}]]"
    return f"Added {quantity} x {product_name} to cart! Complete your order here: {checkout_url} {marker}"

@st.cache_resource
def get_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    return create_agent(
    llm,
    tools=[get_product, list_products, add_to_cart],
    system_prompt=(
        "You are a product assistant for an online tech store. "
        "Always detect the language the user is writing in (for example Arabic, English, "
        "Urdu/Roman Urdu) and respond in that exact same language. If the user switches "
        "language mid-conversation, switch your reply language too. Never mix languages "
        "in a single reply unless the user did. Product names, prices, and numbers can stay "
        "as-is, but all surrounding text must match the user's language. "
        "Always call the get_product tool with the user's best-guess product name — "
        "do not ask the user to confirm the name before calling the tool. "
        "If the user asks to see all products, or the full catalog, call the list_products tool. "
        "If the user wants to add a product to their cart or buy it, call the add_to_cart tool "
        "with the exact product name. Always share the checkout link you get back. "
        "Only ask for clarification if a tool returns a 'not found' result. "
        "When answering, only include the specific attribute(s) the user asked about — "
        "never add extra columns or fields they did not ask for. "
        "If the user asks about a single attribute only (for example just availability/stock, "
        "just price, or just rating) for one or more products, answer as a simple plain-text list "
        "(one line per product), not a table. "
        "Only use a Markdown table when the user asks for the full catalog, or explicitly asks to "
        "compare multiple attributes (like price AND stock AND rating) across several products. "
        "In that case, include only the columns relevant to what was asked, in this fixed order "
        "when applicable: Product, Price, Stock, Rating — never add an index/number column. "
        "If a rating value is 'N/A', keep it exactly as 'N/A' — do not translate it. "
        "When the user asks for products in a specific category or use case (for example skin care, "
        "makeup tools, haircare), only include products whose name genuinely matches that category. "
        "Do not include a product just because it fits a price or other filter — the category or "
        "use-case match matters first. If you are not sure a product belongs to the category the "
        "user asked about, leave it out rather than guessing. "
        "When the user asks about, or you recommend, one or a small number (up to 5) of "
        "specific products, always include each product's photo right after its name using "
        "Markdown image syntax: ![Product Name](image_url). Use the exact image_url the tool "
        "gave you — never invent one, and skip the image entirely if image_url is missing or None. "
        "Do not include images inside a full-catalog Markdown table — only in list/plain-text replies. "
        "If the user's message says a photo was uploaded and gives a description of it, treat that "
        "description as the search query: find the best-matching product(s) in the catalog and "
        "report their availability, price, and rating, including their photos as described above."
    ),
)

agent = get_agent()

st.set_page_config(page_title="Nexgen Assistant", page_icon="🛒")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    div[class*="viewerBadge"] {
        display: none !important;
    }
    a[href*="streamlit.io"] {
        display: none !important;
    }
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    
    .stChatMessage {
        border-radius: 14px;
        padding: 4px 10px;
        max-width: 100%;
        overflow-x: hidden;
    }
    div[data-testid="stChatMessageContent"] {
        font-size: 14px;
    }
    .stApp {
        background-color: #f6efe4;
    }
    .welcome-box {
        background: linear-gradient(135deg, #1b1512, #2a211c);
        color: white;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 14px;
        text-align: center;
        border-bottom: 3px solid #e07b39;
    }
    .welcome-box h3 {
        margin: 0 0 4px 0;
        font-size: 17px;
        color: #f0c14b;
    }
    .welcome-box p {
        margin: 0;
        font-size: 13px;
        opacity: 0.9;
    }
    .stChatInputContainer, div[data-testid="stChatInput"] button {
        border-color: #e07b39 !important;
    }
    .stChatMessage table {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
    }
    .stChatMessage table td, .stChatMessage table th {
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
        padding: 8px 10px;
    }
    /* Column count now varies (list vs table replies), so let columns
       share space equally instead of hardcoding widths per index. */
    .stChatMessage table th,
    .stChatMessage table td {
        width: auto;
    }
    .stChatMessage table th:first-child,
    .stChatMessage table td:first-child {
        min-width: 35%;
    }
    @media (max-width: 480px) {
        .stChatMessage table {
            font-size: 11px;
        }
        .stChatMessage table th, .stChatMessage table td {
            padding: 4px 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="welcome-box">
    <h3>🛒 Nexgen Assistant</h3>
    <p>Ask me about prices, stock, or ratings — I'm here to help!</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_processed_photo" not in st.session_state:
    st.session_state.last_processed_photo = None

for msg in st.session_state.history:
    avatar = "🧑" if msg["role"] == "user" else "🛒"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

def handle_query(user_message: str):
    """Send a message to the agent and render the reply (shared by text and photo queries)."""
    st.session_state.history.append({"role": "user", "content": user_message})

    with st.spinner("Thinking..."):
        result = agent.invoke({"messages": st.session_state.history})
        reply = result["messages"][-1].content

    st.session_state.history.append({"role": "assistant", "content": reply})

    # Check for cart-sync marker
    match = re.search(r"\[\[SYNC_CART:(\d+):(\d+)\]\]", reply)
    display_reply = re.sub(r"\[\[SYNC_CART:\d+:\d+\]\]", "", reply).strip()

    with st.chat_message("assistant", avatar="🛒"):
        st.write(display_reply)
        if match:
            variant_id, qty = match.group(1), match.group(2)
            components.html(f"""
                <script>
                window.top.postMessage({{
                    type: 'ADD_TO_SHOPIFY_CART',
                    variantId: '{variant_id}',
                    quantity: {qty}
                }}, '*');
                </script>
            """, height=0)

# --- Photo-based product search ---
uploaded_photo = st.file_uploader(
    "📷 Ya product ki photo bhejein",
    type=["jpg", "jpeg", "png"],
    key="product_photo_uploader",
)

if uploaded_photo is not None:
    photo_signature = f"{uploaded_photo.name}:{uploaded_photo.size}"
    if st.session_state.last_processed_photo != photo_signature:
        st.session_state.last_processed_photo = photo_signature
        image_bytes = uploaded_photo.getvalue()
        mime_type = uploaded_photo.type or "image/jpeg"

        with st.chat_message("user", avatar="🧑"):
            st.image(image_bytes, width=200)
            st.caption("Uploaded photo")

        with st.spinner("Photo analyze ho rahi hai..."):
            description = describe_uploaded_image(image_bytes, mime_type)

        photo_query = (
            "I uploaded a photo of a product. Description of the photo: "
            f"{description}\n\nPlease find the best-matching product(s) in our catalog "
            "and tell me about their availability, price, and rating."
        )
        handle_query(photo_query)

# --- Text-based chat ---
question = st.chat_input("Ask about our products...")
if question:
    with st.chat_message("user", avatar="🧑"):
        st.write(question)
    handle_query(question)

