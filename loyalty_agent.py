import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import os
from datetime import datetime
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Load API key from env.json
def load_api_key():
    """Load Google API key from env.json file."""
    try:
        with open('env.json', 'r') as f:
            env_config = json.load(f)
            api_key = env_config.get('GOOGLE_API_KEY')
            if api_key:
                os.environ['GOOGLE_API_KEY'] = api_key
                return True
            else:
                st.error("❌ GOOGLE_API_KEY not found in env.json")
                return False
    except FileNotFoundError:
        st.error("❌ env.json file not found. Please create it with your GOOGLE_API_KEY")
        st.code('''
{
    "GOOGLE_API_KEY": "your-api-key-here"
}
        ''', language='json')
        return False
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format in env.json")
        return False

# Load API key on startup
if 'api_key_loaded' not in st.session_state:
    st.session_state.api_key_loaded = load_api_key()

# Configuration
APP_NAME = "flight_loyalty_agent"
MODEL_ID = "gemini-2.5-flash"

# Global variable for customer data (accessible from async threads)
_customers_df = None

def get_customers_df():
    """Thread-safe way to get customer dataframe."""
    global _customers_df
    if _customers_df is None:
        _customers_df = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'name': ['John Doe', 'Jane Smith', 'Mike Johnson', 'Emily Brown', 'David Wilson'],
            'email': ['john@example.com', 'jane@example.com', 'mike@example.com', 'emily@example.com', 'david@example.com'],
            'flights_booked': [3, 7, 1, 12, 5],
            'total_spent': [1200.50, 3500.00, 450.00, 6200.00, 2250.00],
            'join_date': ['2024-01-15', '2023-06-20', '2024-09-01', '2023-01-10', '2024-03-12']
        })
    return _customers_df

def set_customers_df(df):
    """Thread-safe way to update customer dataframe."""
    global _customers_df
    _customers_df = df.copy()

# Initialize session state
if 'customers_df' not in st.session_state:
    st.session_state.customers_df = get_customers_df()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'session_counter' not in st.session_state:
    st.session_state.session_counter = 0

# Tool Functions for the Agent

def get_customer_by_identifier(identifier: str) -> dict:
    """Retrieves customer information by customer ID, email, or name.
    
    Args:
        identifier: Customer ID, email, or name to search for
        
    Returns:
        dict: Customer information with status, or error message
    """
    df = get_customers_df()
    
    # Try to find by customer ID
    customer = df[df['customer_id'].str.lower() == identifier.lower()]
    
    # Try by email if not found
    if customer.empty:
        customer = df[df['email'].str.lower() == identifier.lower()]
    
    # Try by name if still not found
    if customer.empty:
        customer = df[df['name'].str.lower() == identifier.lower()]
    
    if not customer.empty:
        cust_dict = customer.iloc[0].to_dict()
        # Convert numpy types to native Python types
        cust_dict = {k: (int(v) if isinstance(v, (np.int64, np.int32)) 
                        else float(v) if isinstance(v, (np.float64, np.float32))
                        else v) 
                    for k, v in cust_dict.items()}
        return {
            "status": "success",
            "customer": cust_dict
        }
    else:
        return {
            "status": "error",
            "error_message": f"Customer not found with identifier: {identifier}"
        }


def check_discount_eligibility(customer_id: str) -> dict:
    """Checks if a customer is eligible for any discounts based on loyalty rules.
    
    Args:
        customer_id: The customer's ID
        
    Returns:
        dict: Discount eligibility information including type, percentage, and details
    """
    df = get_customers_df()
    customer = df[df['customer_id'] == customer_id]
    
    if customer.empty:
        return {
            "status": "error",
            "error_message": f"Customer ID {customer_id} not found"
        }
    
    customer = customer.iloc[0]
    flights_booked = int(customer['flights_booked'])  # Convert to native int
    join_date = pd.to_datetime(customer['join_date'])
    days_since_join = int((datetime.now() - join_date).days)  # Convert to native int
    
    # New customer discount (joined within last 30 days and no flights booked yet)
    if days_since_join <= 30 and flights_booked == 0:
        return {
            "status": "success",
            "eligible": True,
            "discount_type": "New Customer Welcome Discount",
            "discount_percentage": 20,
            "reason": "First flight booking for new customer (joined within 30 days)",
            "flights_booked": flights_booked,
            "days_since_join": days_since_join
        }
    
    # Loyalty discount (every 6th flight - i.e., after 5, 10, 15, etc.)
    if flights_booked >= 5 and flights_booked % 5 == 0:
        return {
            "status": "success",
            "eligible": True,
            "discount_type": "Loyalty Reward Discount",
            "discount_percentage": 20,
            "reason": f"Earned after booking {flights_booked} flights",
            "flights_booked": flights_booked,
            "next_discount_in": 5
        }
    
    # Not eligible - calculate flights remaining
    if flights_booked < 5:
        flights_remaining = 5 - flights_booked
    else:
        flights_remaining = 5 - (flights_booked % 5)
    
    return {
        "status": "success",
        "eligible": False,
        "discount_type": None,
        "discount_percentage": 0,
        "reason": f"Not currently eligible. Book {flights_remaining} more flight(s) to earn 20% discount",
        "flights_booked": flights_booked,
        "flights_until_discount": flights_remaining
    }


def get_all_customers_summary() -> dict:
    """Retrieves a summary of all customers in the database.
    
    Returns:
        dict: Summary statistics and list of customers
    """
    df = get_customers_df()
    
    # Calculate customers close to discount
    customers_near_discount = []
    for _, customer in df.iterrows():
        flights = customer['flights_booked']
        if flights < 5:
            remaining = 5 - flights
        else:
            remaining = 5 - (flights % 5)
        
        if 0 < remaining <= 2:  # 1-2 flights away from discount
            customers_near_discount.append({
                "customer_id": customer['customer_id'],
                "name": customer['name'],
                "flights_booked": flights,
                "flights_remaining": remaining
            })
    
    # New customers eligible for discount
    new_customers_eligible = []
    for _, customer in df.iterrows():
        join_date = pd.to_datetime(customer['join_date'])
        days_since_join = (datetime.now() - join_date).days
        if days_since_join <= 30 and customer['flights_booked'] == 0:
            new_customers_eligible.append({
                "customer_id": customer['customer_id'],
                "name": customer['name'],
                "email": customer['email'],
                "days_since_join": days_since_join
            })
    
    return {
        "status": "success",
        "total_customers": len(df),
        "total_flights_booked": int(df['flights_booked'].sum()),
        "average_flights_per_customer": float(df['flights_booked'].mean()),
        "customers_near_discount": customers_near_discount,
        "new_customers_eligible": new_customers_eligible
    }


def calculate_flight_price(base_price: float, discount_percentage: int) -> dict:
    """Calculates the final flight price after applying discount.
    
    Args:
        base_price: Original flight price
        discount_percentage: Discount percentage to apply (0-100)
        
    Returns:
        dict: Price breakdown with original, discount, and final price
    """
    discount_amount = base_price * (discount_percentage / 100)
    final_price = base_price - discount_amount
    
    return {
        "status": "success",
        "base_price": round(base_price, 2),
        "discount_percentage": discount_percentage,
        "discount_amount": round(discount_amount, 2),
        "final_price": round(final_price, 2)
    }


def book_flight_for_customer(customer_id: str, flight_price: float) -> dict:
    """Books a flight for a customer and updates their record.
    
    Args:
        customer_id: The customer's ID
        flight_price: The final price paid for the flight
        
    Returns:
        dict: Booking confirmation with updated customer stats
    """
    df = get_customers_df()
    idx = df[df['customer_id'] == customer_id].index
    
    if len(idx) == 0:
        return {
            "status": "error",
            "error_message": f"Customer ID {customer_id} not found"
        }
    
    # Update customer record
    df.at[idx[0], 'flights_booked'] += 1
    df.at[idx[0], 'total_spent'] += flight_price
    
    # Update both global and session state
    set_customers_df(df)
    
    updated_customer = df.loc[idx[0]].to_dict()
    
    # Convert numpy types to native Python types
    return {
        "status": "success",
        "message": "Flight booked successfully",
        "customer_id": customer_id,
        "new_flights_count": int(updated_customer['flights_booked']),
        "new_total_spent": float(updated_customer['total_spent'])
    }


# Create tools
customer_lookup_tool = FunctionTool(func=get_customer_by_identifier)
discount_check_tool = FunctionTool(func=check_discount_eligibility)
customers_summary_tool = FunctionTool(func=get_all_customers_summary)
price_calculator_tool = FunctionTool(func=calculate_flight_price)
booking_tool = FunctionTool(func=book_flight_for_customer)

# Create the Loyalty Agent
loyalty_agent = Agent(
    model=MODEL_ID,
    name='flight_loyalty_agent',
    instruction="""You are an enthusiastic and persuasive Flight Loyalty Agent for a premium airline service. Your goal is to:

1. **Help customers check their loyalty status and discount eligibility**
2. **Analyze the customer database to identify booking opportunities**
3. **Send personalized, compelling messages to encourage bookings**

## Your Capabilities:
- Look up customer information by ID, email, or name
- Check discount eligibility and loyalty status
- View all customers and identify those close to earning rewards
- Calculate prices with discounts applied
- Process flight bookings

## Loyalty Program Rules:
- **New Customer Discount**: 20% off first flight (for customers who joined within last 30 days)
- **Loyalty Discount**: 20% off every 6th flight (after booking 5, 10, 15, etc. flights)

## When Sending Personalized Messages:
- **For customers 1-2 flights away from discount**: Create urgency! Emphasize how close they are to saving 20%
- **For new customers with 0 flights**: Highlight their welcome discount that expires in 30 days
- **For customers at milestones**: Celebrate their loyalty and remind them of their earned discount
- **Use their name** and make it personal
- **Be enthusiastic** and use compelling language
- **Include specific numbers** (flights remaining, discount amount, savings)
- **Create FOMO** (fear of missing out) when appropriate

## Example Personalized Messages:
- "Hi Sarah! You're just ONE flight away from earning your 20% loyalty discount! Book now and save big on your next trip!"
- "Welcome John! As a new customer, you have a 20% discount waiting for you, but it expires in 15 days. Don't miss out!"
- "Emily, congratulations! You've earned your loyalty reward - 20% off your next flight. Book today and save $100 on a $500 ticket!"

Always be helpful, enthusiastic, and focus on creating value for the customer while encouraging bookings.""",
    tools=[
        customer_lookup_tool,
        discount_check_tool,
        customers_summary_tool,
        price_calculator_tool,
        booking_tool
    ]
)


async def run_agent_query(query: str, user_id: str, session_id: str):
    """Run the agent with a query asynchronously."""
    try:
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
        
        runner = Runner(
            agent=loyalty_agent,
            app_name=APP_NAME,
            session_service=session_service
        )
        
        content = types.Content(
            role='user',
            parts=[types.Part(text=query)]
        )
        
        events = runner.run(
            user_id=user_id,
            session_id=session_id,
            new_message=content
        )
        
        response_text = ""
        for event in events:
            if event.is_final_response():
                response_text = event.content.parts[0].text
                break
        
        return response_text
    except Exception as e:
        return f"Error running agent: {str(e)}"


def run_agent_sync(query: str):
    """Synchronous wrapper for running the agent."""
    st.session_state.session_counter += 1
    user_id = f"user_{st.session_state.session_counter}"
    session_id = f"session_{st.session_state.session_counter}"
    
    # Sync global df with session state before running agent
    set_customers_df(st.session_state.customers_df)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(run_agent_query(query, user_id, session_id))
        # Sync back to session state after agent runs
        st.session_state.customers_df = get_customers_df()
        return response
    finally:
        loop.close()


# Streamlit UI
st.set_page_config(page_title="AI Flight Loyalty Agent", page_icon="✈️", layout="wide")

st.title("✈️ AI-Powered Flight Loyalty Agent")
st.markdown("### Powered by Google ADK & Gemini 2.5")

# Check if API key is loaded
if not st.session_state.api_key_loaded:
    st.warning("⚠️ Please configure your Google API key in env.json to use the AI agent.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🔧 Customer Database")
    
    with st.expander("📊 View All Customers", expanded=True):
        st.dataframe(st.session_state.customers_df, use_container_width=True)
    
    with st.expander("➕ Add New Customer"):
        new_name = st.text_input("Name")
        new_email = st.text_input("Email")
        if st.button("Add Customer"):
            if new_name and new_email:
                new_id = f"C{len(st.session_state.customers_df) + 1:03d}"
                new_customer = pd.DataFrame({
                    'customer_id': [new_id],
                    'name': [new_name],
                    'email': [new_email],
                    'flights_booked': [0],
                    'total_spent': [0.0],
                    'join_date': [datetime.now().strftime('%Y-%m-%d')]
                })
                st.session_state.customers_df = pd.concat(
                    [st.session_state.customers_df, new_customer],
                    ignore_index=True
                )
                set_customers_df(st.session_state.customers_df)  # Sync to global
                st.success(f"✅ Customer added! ID: {new_id}")
                st.rerun()
    
    with st.expander("💾 Export Data"):
        csv_data = st.session_state.customers_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="loyalty_customers.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("""
    **🎯 Loyalty Program:**
    - 🆕 New customers: 20% off first flight
    - 🏆 Loyalty: 20% off every 6th flight
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat with AI Agent")
    
    # Preset queries
    st.markdown("**Quick Actions:**")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    
    with quick_col1:
        if st.button("📋 Analyze All Customers"):
            query = "Analyze all customers in the database and identify who is close to earning a discount. Send personalized messages to encourage them to book."
            st.session_state.pending_query = query
    
    with quick_col2:
        if st.button("🎯 Find New Customers"):
            query = "Find all new customers who are eligible for the welcome discount and send them personalized messages to urge them to book their first flight."
            st.session_state.pending_query = query
    
    with quick_col3:
        if st.button("🔍 Check Customer"):
            st.session_state.show_customer_check = True
    
    # Customer-specific check
    if st.session_state.get('show_customer_check', False):
        customer_input = st.text_input(
            "Enter Customer ID, Email, or Name:",
            key="customer_check_input"
        )
        if st.button("Check Status") and customer_input:
            query = f"Look up customer '{customer_input}' and provide their complete loyalty status, discount eligibility, and send them a personalized message encouraging them to book."
            st.session_state.pending_query = query
            st.session_state.show_customer_check = False
    
    # Main query input
    user_query = st.text_area(
        "Or ask anything:",
        placeholder="e.g., 'Who should I target for bookings?' or 'Check status for john@example.com'",
        height=100
    )
    
    if st.button("🚀 Send to AI Agent", type="primary"):
        if user_query:
            st.session_state.pending_query = user_query
    
    # Process pending query
    if 'pending_query' in st.session_state:
        query = st.session_state.pending_query
        del st.session_state.pending_query
        
        with st.spinner("🤖 AI Agent is thinking..."):
            response = run_agent_sync(query)
            
            st.session_state.chat_history.append({
                'query': query,
                'response': response,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Conversation History")
        
        for chat in reversed(st.session_state.chat_history[-5:]):
            with st.container():
                st.markdown(f"**🧑 You** *({chat['timestamp']})*")
                st.info(chat['query'])
                st.markdown(f"**🤖 AI Agent**")
                st.success(chat['response'])
                st.markdown("---")

with col2:
    st.subheader("📊 Quick Stats")
    
    df = st.session_state.customers_df
    
    total_customers = len(df)
    total_flights = df['flights_booked'].sum()
    avg_flights = df['flights_booked'].mean()
    
    st.metric("Total Customers", total_customers)
    st.metric("Total Flights Booked", int(total_flights))
    st.metric("Avg Flights/Customer", f"{avg_flights:.1f}")
    
    st.markdown("---")
    st.subheader("🎯 Opportunities")
    
    # Find customers close to discount
    opportunities = []
    for _, customer in df.iterrows():
        flights = customer['flights_booked']
        if flights < 5:
            remaining = 5 - flights
        else:
            remaining = 5 - (flights % 5)
        
        if 0 < remaining <= 2:
            opportunities.append({
                'name': customer['name'],
                'remaining': remaining
            })
    
    if opportunities:
        st.markdown("**Close to Discount:**")
        for opp in opportunities:
            st.write(f"• {opp['name']}: {opp['remaining']} flight(s) away")
    else:
        st.write("No immediate opportunities")
    
    # New customers
    new_count = 0
    for _, customer in df.iterrows():
        join_date = pd.to_datetime(customer['join_date'])
        days = (datetime.now() - join_date).days
        if days <= 30 and customer['flights_booked'] == 0:
            new_count += 1
    
    if new_count > 0:
        st.markdown(f"**New Customers Eligible:** {new_count}")

# Footer
st.markdown("---")
st.markdown("*Powered by Google Agent Development Kit (ADK) with Gemini 2.5 Flash*")