# ✈️ AI-Powered Flight Loyalty Agent

An intelligent customer loyalty management system powered by Google's Agent Development Kit (ADK) and Gemini 2.5 Flash. This application helps airlines manage customer loyalty programs, check discount eligibility, and send personalized booking encouragement messages.

## 🌟 Features

- **AI-Powered Agent**: Uses Google Gemini 2.5 to intelligently interact with customer data
- **Customer Management**: Add, view, and manage customer information
- **Loyalty Program Automation**: 
  - 20% discount for new customers (first 30 days)
  - 20% discount every 6th flight (after 5, 10, 15 flights, etc.)
- **Personalized Messaging**: AI generates compelling, personalized messages to encourage bookings
- **Opportunity Detection**: Automatically identifies customers close to earning discounts
- **Flight Booking**: Process bookings and update customer records in real-time
- **Data Export**: Download customer data as CSV

## 🛠️ Prerequisites

- Python 3.8 or higher
- Google API Key (with access to Gemini models)
- pip package manager

## 📦 Installation

### Step 1: Clone or Download the Project

```bash
git clone <your-repository-url>
cd my-agent-project
```

Or download and extract the ZIP file.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `google-genai` - Google Generative AI SDK
- `google-adk` - Google Agent Development Kit
- `python-dateutil` - Date utilities

### Step 3: Configure Google API Key

1. **Get your Google API Key:**
   - Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the generated API key

2. **Create `env.json` file:**

Create a file named `env.json` in the project root directory with the following content:

```json
{
    "GOOGLE_API_KEY": "your-actual-api-key-here"
}
```

Replace `your-actual-api-key-here` with your actual API key.

⚠️ **Important**: Never commit `env.json` to version control. Add it to `.gitignore`:

```bash
echo "env.json" >> .gitignore
```

### Step 4: Run the Application

```bash
streamlit run loyalty_agent.py
```

Replace `loyalty_agent.py` with your actual filename if different.

The application will open in your default browser at `http://localhost:8501`

## 🚀 Usage Guide

### Quick Actions

The application provides three quick action buttons:

1. **📋 Analyze All Customers**: AI analyzes the entire customer database and identifies booking opportunities
2. **🎯 Find New Customers**: Targets new customers eligible for welcome discounts
3. **🔍 Check Customer**: Look up a specific customer by ID, email, or name

### Customer Database Management

**View Customers**: 
- Expand "📊 View All Customers" in the sidebar to see all customer records

**Add New Customer**:
1. Click "➕ Add New Customer" in the sidebar
2. Enter customer name and email
3. Click "Add Customer"
4. New customer gets a unique ID and is added with 0 flights booked

**Export Data**:
- Click "💾 Export Data" in the sidebar
- Download the customer database as CSV

### Interacting with the AI Agent

**Preset Queries**: Use the quick action buttons for common tasks

**Custom Queries**: Type your own questions in the text area, such as:
- "Who should I target for bookings?"
- "Check status for john@example.com"
- "Find customers who are 1 flight away from earning a discount"
- "Book a $500 flight for customer C001"
- "Calculate price for a $600 flight with 20% discount"

**AI Capabilities**:
- Look up customer information by ID, email, or name
- Check discount eligibility and loyalty status
- Analyze all customers and identify opportunities
- Calculate prices with discounts
- Process flight bookings and update records
- Generate personalized marketing messages

## 📊 Loyalty Program Rules

### New Customer Discount
- **Eligibility**: Customers who joined within the last 30 days and have 0 flights booked
- **Discount**: 20% off their first flight
- **Purpose**: Welcome incentive for new customers

### Loyalty Reward Discount
- **Eligibility**: Every 6th flight (after booking 5, 10, 15, 20, etc. flights)
- **Discount**: 20% off
- **Purpose**: Reward repeat customers for loyalty

## 🏗️ Project Structure

```
flight-loyalty-agent/
│
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── env.json               # API key configuration (create this)
├── README.md              # This file
└── .gitignore            # Git ignore file
```

## 🔧 Troubleshooting

### "API key not found" error
- Ensure `env.json` exists in the project root
- Verify the JSON format is correct
- Check that your API key is valid

### "Module not found" errors
- Run `pip install -r requirements.txt` again
- Ensure you're using Python 3.8+
- Try creating a virtual environment

### Serialization errors (numpy.int64)
- This is fixed in the updated code
- Ensure you're using the latest version with native Python type conversions

### Agent not responding
- Check your internet connection
- Verify your API key has access to Gemini models
- Check Google AI Studio for API quota limits

## 🔐 Security Notes

- **Never commit `env.json`** to version control
- Keep your API key secure and private
- Use environment variables in production deployments
- Monitor API usage to avoid unexpected costs

## 📝 Example Workflows

### Workflow 1: Target Customers Near Discount
1. Click "📋 Analyze All Customers"
2. AI identifies customers 1-2 flights away from earning rewards
3. Review personalized messages generated by AI
4. Use insights for marketing campaigns

### Workflow 2: Welcome New Customers
1. Click "🎯 Find New Customers"
2. AI finds customers with welcome discount eligibility
3. AI generates urgency-driven messages highlighting the 30-day expiration
4. Encourage first bookings

### Workflow 3: Process a Booking
1. Click "🔍 Check Customer"
2. Enter customer identifier
3. AI shows discount eligibility
4. Ask AI to "Book a $500 flight for customer C001"
5. System updates records automatically

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

[Add your license here]

## 🙋 Support

For issues related to:
- **Google ADK/API**: Check [Google AI Documentation](https://docs.claude.com)
- **Application bugs**: Open an issue in this repository
- **Feature requests**: Open an issue with the "enhancement" label

## 🎯 Future Enhancements

- [ ] Database persistence (SQLite/PostgreSQL)
- [ ] Email integration for automated messaging
- [ ] Advanced analytics dashboard
- [ ] Multi-tier loyalty programs
- [ ] Integration with booking systems
- [ ] Customer segmentation
- [ ] A/B testing for messaging

---

**Built with ❤️ using Google ADK and Gemini 2.5 Flash**