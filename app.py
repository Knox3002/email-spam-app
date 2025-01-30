import streamlit as st
import pickle

model = prickle.load(open('model.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

st.title("email spam classification application")
st.write("this is a machine learning application to classify email as spam or ham")
user_input = st.text_area("enetr an email to classify", height= 150)
if st.button("classify") :
    if user_input:
        data=[user_input]
        vectorized_data=cv.transform(data).toarray()
        result=model.predict(vectorized_data)
        if result[0]==0:
            st.write("the email is not spam")
        else:
            st.write("the email is spam")
    else:
        st.write("please type wmail to classify")
