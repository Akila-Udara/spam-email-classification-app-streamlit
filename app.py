import streamlit as st
import joblib

@st.cache_data()
def load_model():
    # - Loading the spam_model
    model = joblib.load('spam_model.pkl')
    
    # - Loading the tfidf_vectorizer
    feature_extraction = joblib.load('tfidf_vectorizer.pkl')
    
    return model, feature_extraction

def trained_model(input_mail, model, feature_extraction):
    input_mail_features = feature_extraction.transform(input_mail)

    prediction = model.predict(input_mail_features)

    if prediction[0] >= 0.5:
        image_path = 'spam.png'
    else:
        image_path = 'ham.png'

    return image_path

# Streamlit App
def main():
    
    # CSS Styling
    st.markdown(
        """
        <style>
        .centered-heading {
            text-align: center;
            padding-bottom: 40px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 class='centered-heading'>Spam Email Classification System</h1>", unsafe_allow_html=True)

    # Loading the model (1 time run at the beginning)
    model, feature_extraction = load_model()

    # User input
    input_mail = st.text_input('**Enter the Suspect Mail:**')
    input_mail = [input_mail]

    # Display the results
    if st.button('Predict'):
        image_path = trained_model(input_mail, model, feature_extraction)
        st.image(image_path, use_column_width=True)

if __name__ == '__main__':
    main()
