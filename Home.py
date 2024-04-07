import streamlit as st

option = st.selectbox(
    ":violet[당신이 좋아하는 자동차는? :car:]",
    ("람보르기니", "부가티", "페라리"),
    index=1,
    placeholder="좋아하는 자동차를 선택하세요.",
)

if option:
    st.write(f"당신이 좋아하는 자동차는 {option}입니다.")
else:
    st.write("좋아하는 자동차를 선택해주세요.")
