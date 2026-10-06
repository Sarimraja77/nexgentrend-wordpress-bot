import streamlit as st
import streamlit.components.v1 as components
import re
import os
import base64
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

# WooCommerce Credentials & URL
STORE_URL = os.getenv("WC_STORE_URL", "https://pk.nexgentrend.com")
CK = os.getenv("WC_CONSUMER_KEY")
CS = os.getenv("WC_CONSUMER_SECRET")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

st.set_page_config(page_title="Nexgen Assistant", page_icon="🛒")

# --- Sidebar Gradient Theme Selector ---
st.sidebar.title("⚙️ Settings")
theme_option = st.sidebar.selectbox(
    "Choose Theme Style:",
    ["Option 2: Solid Dark Mode (#121212)", "Gradient Dark Mode (Decent & Premium)"]
)

if "Solid Dark" in theme_option:
    bg_color = "#121212"
    text_color = "#f0f0f0"
    welcome_bg = "linear-gradient(135deg, #1b1512, #2a211c)"
else:
    # Decent dark gradient background styling
    bg_color = "linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%)"
    text_color = "#f8fafc"
    welcome_bg = "linear-gradient(135deg, #1e293b, #334155)"

def _clean_product_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _normalize_content(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "message"):
            if key in content:
                text = _normalize_content(content[key])
                if text:
                    return text
        return str(content)
    return str(content)

def _to_langchain_message(message: dict[str, Any]):
    role = str(message.get("role", "user")).lower()
    content = _normalize_content(message.get("content", ""))
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)

def fetch_products():
    """Fetch live products from WooCommerce REST API."""
    if not CK or not CS:
        return {}

    url = f"{STORE_URL}/wp-json/wc/v3/products"
    auth = (CK, CS)

    try:
        response = requests.get(url, auth=auth, timeout=20)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"WooCommerce Fetch Error: {e}")
        return {}

    if not isinstance(data, list):
        return {}

    products = {}
    for item in data:
        if not isinstance(item, dict):
            continue

        name = _clean_product_name(item.get("name"))
        if not name:
            continue

        price = item.get("price", "0")
        currency = "PKR"

        stock_qty = item.get("stock_quantity")
        stock = stock_qty if stock_qty is not None else (item.get("stock_status") or "In Stock")

        rating = item.get("average_rating", "N/A")
        description = item.get("description", "")
        product_id = item.get("id")

        images = item.get("images", [])
        image_url = images[0].get("src") if isinstance(images, list) and images and isinstance(images[0], dict) else None

        products[name] = {
            "Price": price,
            "Currency": currency,
            "Stock": stock,
            "Rating": rating,
            "description": description,
            "variant_id": product_id,
            "image_url": image_url,
        }
    return products

@st.cache_data(ttl=300)
def get_products():
    return fetch_products()

def describe_uploaded_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Use Groq's vision model to describe a user-uploaded product photo."""
    if not image_bytes:
        return ""

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

    try:
        response = vision_llm.invoke([message])
        return _normalize_content(response.content)
    except Exception:
        return ""

@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, stock, rating, description and image URL."""
    if not isinstance(name, str):
        name = str(name or "")

    products = get_products()
    if not products:
        return "No products are currently available or API credentials are missing."

    products_lookup = {k.lower(): v for k, v in products.items()}
    p = products_lookup.get(name.strip().lower())
    if not p:
        return f"product not found. Available products: {', '.join(products.keys())}"
    return str(p)

@tool
def list_products() -> str:
    """List all available products with their prices, stock, rating and image URL."""
    products = get_products()
    if not products:
        return "No products are currently available or API credentials are missing."

    lines = []
    for name, info in products.items():
        lines.append(
            f"- **{name}**: {info['Price']} {info['Currency']} | Stock: {info['Stock']} | "
            f"Rating: {info['Rating']} | Image: {info['image_url']}"
        )
    return "\n".join(lines)

@tool
def add_to_cart(product_name: str, quantity: int = 1) -> str:
    """Add a product to the cart by name and return a checkout link."""
    if not isinstance(product_name, str):
        product_name = str(product_name or "")

    cleaned_name = product_name.strip()
    if not cleaned_name:
        return "Please provide a valid product name."

    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        return "Quantity must be a number."

    if quantity <= 0:
        return "Quantity must be greater than zero."

    products = get_products()
    if not products:
        return "No products are currently available."

    products_lookup = {k.lower(): v for k, v in products.items()}
    p = products_lookup.get(cleaned_name.lower())
    if not p:
        return f"Product not found. Available products: {', '.join(products.keys())}"
    if not p.get("variant_id"):
        return "Sorry, this product cannot be added to cart right now."

    product_id = p["variant_id"]
    checkout_url = f"{STORE_URL}/checkout/?add-to-cart={product_id}&quantity={quantity}"

    marker = f"[[SYNC_CART:{product_id}:{quantity}]]"
    return f"Added {quantity} x {cleaned_name} to cart! Complete your order here: {checkout_url} {marker}"

@st.cache_resource
def get_agent():
    llm = ChatGroq(model=CHAT_MODEL, temperature=0)
    return create_agent(
        llm,
        tools=[get_product, list_products, add_to_cart],
        system_prompt=(
            "You are a helpful product assistant for an online tech store (NexgenTrend). "
            "Always detect the language the user is writing in and respond in that exact same language. "
            "If the user asks to see all products, or the full catalog, you MUST call the list_products tool and present the results clearly. "
            "If the user wants to look up a specific product, call the get_product tool. "
            "If the user wants to add a product to their cart, call the add_to_cart tool."
        ),
    )

agent = get_agent()

# Dynamic Styling with Gradient Background
st.markdown(f"""
<style>
    #MainMenu, footer, header {{visibility: hidden;}}
    div[class*="viewerBadge"] {{ display: none !important; }}
    a[href*="streamlit.io"] {{ display: none !important; }}
    .block-container {{padding-top: 1rem; padding-bottom: 1rem;}}
    
    .stApp {{
        background: {bg_color};
        background-attachment: fixed;
        color: {text_color};
    }}
    .welcome-box {{
        background: {welcome_bg};
        color: white;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 14px;
        text-align: center;
        border-bottom: 3px solid #e07b39;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .welcome-box h3 {{
        margin: 0 0 4px 0;
        font-size: 17px;
        color: #f0c14b;
    }}
    .welcome-box p {{
        margin: 0;
        font-size: 13px;
        opacity: 0.9;
    }}
    .stChatInputContainer, div[data-testid="stChatInput"] button {{
        border-color: #e07b39 !important;
    }}
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
    """Send a message to the agent and render the reply with safe error fallback."""
    if not user_message or not user_message.strip():
        return

    st.session_state.history.append({"role": "user", "content": user_message})

    try:
        with st.spinner("Thinking..."):
            message_history = [_to_langchain_message(message) for message in st.session_state.history]
            result = agent.invoke({"messages": message_history})
            
            # Safe extraction of response from various agent return formats
            if isinstance(result, dict) and "messages" in result:
                last_msg = result["messages"][-1]
                reply = getattr(last_msg, "content", str(last_msg))
            else:
                reply = str(result)
    except Exception as e:
        # Fallback to direct products list if tool invocation encounters issues
        if "list" in user_message.lower() or "all products" in user_message.lower():
            prods = get_products()
            if prods:
                reply = "Here are our available products:\n" + "\n".join([f"- **{k}**: {v['Price']} PKR" for k, v in prods.items()])
            else:
                reply = "Could not fetch products right now. Please check WooCommerce API credentials."
        else:
            reply = f"I encountered a technical glitch, but I'm here! Error details: {str(e)}"

    reply = _normalize_content(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})

    match = re.search(r"\[\[SYNC_CART:(\d+):(\d+)\]\]", reply)
    display_reply = re.sub(r"\[\[SYNC_CART:\d+:\d+\]\]", "", reply).strip()

    with st.chat_message("assistant", avatar="🛒"):
        st.write(display_reply)
        if match:
            product_id, qty = match.group(1), match.group(2)
            components.html(f"""
                <script>
                window.top.postMessage({{
                    type: 'ADD_TO_WOOCOMMERCE_CART',
                    productId: '{product_id}',
                    quantity: {qty}
                }}, '*');
                </script>
            """, height=0)

# --- Photo-based product search ---
uploaded_photo = st.file_uploader(
    "📷 Search Products By Photo",
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

        if description:
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