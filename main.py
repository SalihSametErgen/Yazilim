import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Panel")
app_mode = st.sidebar.selectbox("Sayflar",["Ana Sayfa","Hakkında","Hastalık Tanımlama"])

#Main Page
if(app_mode=="Ana Sayfa"):
    st.header("BİTKİ HASTALIKLARI TANIMLAMA SİSTEMİ")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Bitki Hastalığı Tanımlama Sistemimize Hoşgeldin! 🌿🔍
    
    Amacımız bitki hastalıklarını etkili bir şekilde tespit etmeye yardımcı olmaktır. Bir bitkinin görüntüsünü yükleyin ve sistemimiz herhangi bir hastalık belirtisini 
    tespit etmek için onu analiz edecektir. Birlikte, ürünlerimizi koruyalım ve daha sağlıklı bir hasat sağlayalım!

    ### Nasıl Çalışır
    1. **Görüntü Yükle:** **Hastalık Tanıma** sayfasına gidin ve şüpheli hastalıkları olan bir bitkinin görüntüsünü yükleyin.
    2. **Analiz:** Sistemimiz olası hastalıkları tespit etmek için görüntüyü gelişmiş algoritmalar kullanarak işleyecektir.
    3. **Sonuçlar:** Daha fazla işlem için sonuçları ve önerileri görüntüleyin.

    ### Neden Bizi Seçmelisiniz?
    - **Doğruluk:** Sistemimiz doğru hastalık tespiti için son teknoloji makine öğrenimi tekniklerini kullanır.
    - **Kullanıcı Dostu:** Sorunsuz kullanıcı deneyimi için basit ve sezgisel arayüz.
    - **Hızlı ve Verimli:** Sonuçları saniyeler içinde alın, hızlı karar almaya olanak tanır.

    ### Başlayın
    Kenar çubuğundaki **Hastalık Tanıma** sayfasına tıklayarak bir resim yükleyin ve Bitki Hastalığı Tanıma Sistemimizin gücünü deneyimleyin!

    ### Hakkımızda
    Proje, ekibimiz ve hedeflerimiz hakkında daha fazla bilgi edinmek için **Hakkımızda** sayfasını ziyaret edin.
    """)

#Hakkında Kısmı
elif(app_mode=="Hakkında"):
    st.header("Hakkında")
    st.markdown("""
                #### Veri Seti Hakkında
                Bu veri seti, orijinal veri setinden çevrimdışı çoğaltma kullanılarak yeniden oluşturulmuştur. Orijinal veri seti bu github deposunda bulunabilir.
                Bu veri seti, 38 farklı sınıfa ayrılmış sağlıklı ve hastalıklı mahsul yapraklarının yaklaşık 87K rgb görüntüsünden oluşur. Toplam veri seti, dizin yapısını koruyarak 80/20 oranında eğitim ve doğrulama kümesine bölünmüştür.
                Daha sonra tahmin amacıyla 33 test görüntüsü içeren yeni bir dizin oluşturulur.
                #### İçerik
                1. train (70295 görüntü)
                2. test (33 görüntü)
                3. validation (17572 görüntü)

                """)

#Prediction Page
elif(app_mode=="Hastalık Tanımlama"):
    st.header("Hastalık Tanımlama")
    test_image = st.file_uploader("Resim Seç:")
    if(st.button("Görseli Göster")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Tahmin")):
        st.snow()
        st.write("Sistemin Tahmini")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Sistemimizin tahmini budur: {}".format(class_name[result_index]))

# 'Elma___Elma_kabuk_lezyonu', 'Elma___Siyah_çürüklük', 'Elma___Sedir_elma_pası', 'Elma___sağlıklı',
# 'Yaban_mersini___sağlıklı', 'Kiraz_(ekşi_dahil)___Külleme', 
# 'Kiraz_(ekşi_dahil)___sağlıklı', 'Mısır_(koçan)___Cercospora_yaprak_lekesi Gri_yaprak_lekesi', 
# 'Mısır_(koçan)___Yaygın_pası', 'Mısır_(koçan)___Kuzey_Yaprak_Yanıklığı', 'Mısır_(koçan)___sağlıklı', 
# 'Üzüm___Siyah_çürüklük', 'Üzüm___Esca_(Siyah_Çiçeklenme)', 'Üzüm___Yaprak_lekesi_(Isariopsis_Yaprak_Lekesi)', 
# 'Üzüm___sağlıklı', 'Portakal___Haunglongbing_(Narenciye_yeşillenmesi)', 'Şeftali___Bakteriyel_leke',
# 'Şeftali___sağlıklı', 'Dolmalık_biber___Bakteriyel_leke', 'Dolmalık_biber___sağlıklı', 
# 'Patates___Erken_yanıklık', 'Patates___Geç_yanıklık', 'Patates___sağlıklı', 
# 'Ahududu___sağlıklı', 'Soya_fasulyesi___sağlıklı', 'Kabak___Külleme', 
# 'Çilek___Yaprak_yanıklığı', 'Çilek___sağlıklı', 'Domates___Bakteriyel_leke', 
# 'Domates___Erken_yanıklık', 'Domates___Geç_yanıklık', 'Domates___Yaprak_Küfü', 
# 'Domates___Septoria_yaprak_lekesi', 'Domates___Örümcek_akarları İki_noktalı_örümcek_akarı', 
# 'Domates___Hedef_Leke', 'Domates___Domates_Sarı_Yaprak_Kıvrım_Virüsü', 'Domates___Domates_mozaik_virüsü',
# 'Domates___sağlıklı'
