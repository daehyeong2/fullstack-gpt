import streamlit as st
import os

folder = "pages"
files = ["DocumentGPT", "PrivateGPT", "QuizGPT", "SiteGPT", "MeetingGPT", "InvestorGPT"]

# Create files in pages folder
os.makedirs(folder, exist_ok=True)

for idx, filename in enumerate(files, 1):
    file = os.path.join(folder, f"{str(idx).zfill(2)}_{filename}.py")
    if os.path.exists(file):
        continue
    with open(file, "w") as f:
        f.write(f"import streamlit as st\n\nst.title('{filename}')")


st.set_page_config(page_title="FullstackGPT Home", page_icon="🪄")

st.title("FullstackGPT Home")

st.markdown(
    """
- [x] [DocumentGPT](/DocumentGPT)
- [x] [PrivateGPT](/PrivateGPT)
- [ ] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)
