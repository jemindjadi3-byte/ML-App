import streamlit as st

# Page title
st.title("My Simple Streamlit App")

# Sidebar controls
st.sidebar.header("Controls")
name = st.sidebar.text_input("Enter your name:", "Student")
value = st.sidebar.slider("Choose a number:", 1, 100, 50)

# Main content
st.write(f"### Hello, {name} 👋")
st.write("This is a simple example Streamlit app.")

st.write(f"You selected the number **{value}**.")

# Button example
if st.button("Click me"):
    st.success("Button clicked!")

# Example chart
import pandas as pd
import numpy as np

data = pd.DataFrame({
    "x": np.arange(10),
    "y": np.random.randn(10)
})

st.line_chart(data, x="x", y="y")


