import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import yfinance as yf

# Set page config
st.set_page_config(page_title="Financial Companion", layout="wide")

# ===== DATA SECTION =====
sentiment_data = {
    'text': [
        # Netral
        'Apa kabar?', 'Bagaimana perasaanmu?', 'Kamu baik-baik saja?', 'Hari ini biasa saja', 'Tidak ada yang spesial hari ini',
        # Peduli
        'Kamu sudah makan?', 'Semoga harimu menyenangkan', 'Jangan lupa istirahat', 'Bagaimana kabarmu hari ini?',
        # Positif
        'Kamu terlihat bahagia', 'Aku merasa optimis hari ini', 'Semangat menjalani hari!', 'Aku yakin semuanya akan baik-baik saja',
        # Sedih
        'Apakah kamu sedang sedih?', 'Aku merasa sedih hari ini', 'Hari ini terasa berat', 'Aku kehilangan semangat',
        # Senang
        'Aku sangat senang hari ini!', 'Aku bahagia sekali', 'Hari ini menyenangkan', 'Aku mendapatkan kabar baik',
        # Tertekan
        'Aku merasa tertekan', 'Banyak tekanan di pekerjaan', 'Aku stres dengan tugas-tugas', 'Beban pikiran terasa berat',
        # Antusias
        'Hari ini luar biasa!', 'Aku sangat antusias', 'Tidak sabar menunggu besok', 'Aku bersemangat mencoba hal baru',
        # Kecewa
        'Aku kecewa dengan hasilnya', 'Hasilnya tidak sesuai harapan', 'Aku merasa gagal', 'Kecewa dengan keputusan ini',
        # Bangga
        'Saya bangga dengan pencapaianmu', 'Aku bangga pada diriku sendiri', 'Pencapaian ini luar biasa', 'Aku berhasil melewati tantangan',
        # Sedih (tambahan)
        'Ini sangat menyedihkan', 'Aku merasa sendiri', 'Aku ingin menangis',
        # Takut
        'Aku takut tentang masa depan', 'Aku khawatir dengan situasi ini', 'Aku merasa cemas', 'Takut menghadapi kenyataan',
        # Bahagia
        'Kamu membuatku bahagia', 'Aku merasa sangat bahagia', 'Hari ini penuh kebahagiaan', 'Aku tersenyum sepanjang hari',
        # Marah
        'Aku marah dengan situasi ini', 'Aku kesal sekali', 'Aku tidak suka dengan ini', 'Aku merasa emosi',
        # Bersyukur
        'Aku merasa bersyukur hari ini', 'Aku berterima kasih atas segalanya', 'Bersyukur dengan apa yang dimiliki', 'Aku menghargai hidupku'
    ],
    'sentiment': [
        # Netral
        'netral', 'netral', 'netral', 'netral', 'netral',
        # Peduli
        'peduli', 'peduli', 'peduli', 'peduli',
        # Positif
        'positif', 'positif', 'positif', 'positif',
        # Sedih
        'sedih', 'sedih', 'sedih', 'sedih',
        # Senang
        'senang', 'senang', 'senang', 'senang',
        # Tertekan
        'tertekan', 'tertekan', 'tertekan', 'tertekan',
        # Antusias
        'antusias', 'antusias', 'antusias', 'antusias',
        # Kecewa
        'kecewa', 'kecewa', 'kecewa', 'kecewa',
        # Bangga
        'bangga', 'bangga', 'bangga', 'bangga',
        # Sedih (tambahan)
        'sedih', 'sedih', 'sedih',
        # Takut
        'takut', 'takut', 'takut', 'takut',
        # Bahagia
        'bahagia', 'bahagia', 'bahagia', 'bahagia',
        # Marah
        'marah', 'marah', 'marah', 'marah',
        # Bersyukur
        'bersyukur', 'bersyukur', 'bersyukur', 'bersyukur'
    ]
}

# Enhanced responses for each sentiment
sentiment_responses = {
    'netral': "Saya baik-baik saja, terima kasih sudah bertanya.",
    'peduli': "Terima kasih sudah peduli. Saya menghargai perhatianmu.",
    'positif': "Senang mendengarnya! Semoga harimu terus menyenangkan!",
    'sedih': "Saya turut prihatin. Jika butuh teman bicara, saya di sini untukmu.",
    'senang': "Wah! Senang sekali mendengar kabar baik ini! 🎉",
    'tertekan': "Saya mengerti perasaanmu. Ingat, setiap masalah pasti ada jalan keluarnya.",
    'antusias': "Energi positifmu sangat menginspirasi! ✨",
    'kecewa': "Saya mengerti kekecewaanmu. Semoga ada hikmah di balik ini semua.",
    'bangga': "Kamu pantas merasa bangga! Ini pencapaian yang luar biasa!",
    'takut': "Tidak apa merasa takut. Mari kita bicarakan apa yang mengkhawatirkanmu.",
    'bahagia': "Kebahagiaanmu membuat saya ikut senang! 😊",
    'marah': "Saya mengerti kemarahanmu. Cobalah tarik napas dalam-dalam.",
    'bersyukur': "Sikap bersyukur adalah kunci kebahagiaan. Bagus sekali! 🙏"
}

# Stock data and tips
stock_tips = [
    "Analisis fundamental: Pelajari laporan keuangan perusahaan sebelum berinvestasi",
    "Diversifikasi portofolio: Jangan taruh semua telur dalam satu keranjang",
    "Investasi jangka panjang cenderung lebih stabil daripada trading harian",
    "Pahami risiko: Harga saham bisa naik dan turun",
    "Gunakan analisis teknikal untuk identifikasi tren pasar",
    "Investasi rutin (dollar-cost averaging) bisa mengurangi risiko timing pasar",
    "Jangan biarkan emosi mempengaruhi keputusan investasi Anda",
    "Pelajari tentang PER (Price to Earning Ratio) untuk menilai valuasi saham"
]

# ===== MODEL SECTION =====
@st.cache_resource
def train_sentiment_model():
    df = pd.DataFrame(sentiment_data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    model = SVC(kernel='linear')
    model.fit(X, y)
    return model, vectorizer

model, vectorizer = train_sentiment_model()

# ===== STOCK DATA FUNCTION =====
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = ((current_price - prev_close) / prev_close) * 100
            return {
                'price': round(current_price, 2),
                'change': round(change, 2),
                'currency': stock.info.get('currency', 'USD'),
                'name': stock.info.get('longName', ticker)
            }
    except:
        return None

# ===== SIDEBAR SECTION =====
with st.sidebar:
    st.title("💰 Alat Keuangan")
    
    # Stock Market Tools
    st.subheader("📈 Informasi Saham")
    stock_query = st.text_input("Cari saham (contoh: BBCA.JK):")
    
    if st.button("Dapatkan Update Saham"):
        if stock_query:
            with st.spinner("Mengambil data..."):
                stock_data = get_stock_data(stock_query)
                if stock_data:
                    change_sign = "+" if stock_data['change'] >= 0 else ""
                    st.success(
                        f"**{stock_data['name']}**\n\n"
                        f"Harga: {stock_data['currency']}{stock_data['price']} "
                        f"({change_sign}{stock_data['change']}%)\n\n"
                        f"*Data mungkin tertunda 15-20 menit*"
                    )
                else:
                    st.error("Saham tidak ditemukan. Cek kode saham.")
        else:
            st.warning("Masukkan kode saham terlebih dahulu")
    
    # Stock Tips Section
    st.markdown("---")
    st.subheader("💡 Tips Saham Harian")
    tip = random.choice(stock_tips)
    st.info(tip)
    if st.button("Tips Lainnya"):
        st.rerun()

# ===== MAIN CONTENT =====
st.title("Financial Companion")
st.markdown("Selamat datang di Financial Companion! Aplikasi ini dirancang untuk membantu Anda dengan analisis keuangan dan memberikan inspirasi harian yang positif.") 

tab1, tab2 = st.tabs(["Analisis Sentimen", "Rekomendasi Harian"])

with tab1:
    st.header("💬 Analisis Emosi")
    user_input = st.text_area("Masukkan pesan atau perasaan Anda:", "Apa kabar?")
    
    if st.button("Analisis Sekarang"):
        if user_input:
            with st.spinner("Menganalisis emosi..."):
                vec = vectorizer.transform([user_input])
                sentiment = model.predict(vec)[0]
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Hasil Analisis**")
                    st.metric("Emosi Terdeteksi", sentiment.capitalize())
                with col2:
                    st.markdown("**Respons**")
                    st.success(sentiment_responses.get(sentiment, "Terima kasih sudah berbagi perasaan."))
        else:
            st.warning("Masukkan teks terlebih dahulu")

with tab2:
    st.header("✨ Inspirasi Harian")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎶 Lagu Inspiratif")
        if 'song' not in st.session_state:
            st.session_state.song = random.choice([
                "Pharell Williams - Happy",
                "Bob Marley - Three Little Birds",
                "The Beatles - Here Comes the Sun"
            ])
        st.write(f"**{st.session_state.song}**")
        if st.button("Ganti Lagu"):
            st.session_state.song = random.choice([
                "Pharell Williams - Happy",
                "Bob Marley - Three Little Birds",
                "The Beatles - Here Comes the Sun"
            ])
            st.rerun()
    
    with col2:
        st.subheader("💬 Kutipan Motivasi")
        if 'quote' not in st.session_state:
            st.session_state.quote = random.choice([
                {"quote": "The stock market is filled with individuals who know the price of everything, but the value of nothing.", "author": "Philip Fisher"},
                {"quote": "Investing should be more like watching paint dry or watching grass grow.", "author": "Paul Samuelson"},
                {"quote": "The most important quality for an investor is temperament, not intellect.", "author": "Warren Buffett"},
                {"quote": "Know what you own, and know why you own it.", "author": "Peter Lynch"},
                {"quote": "The individual investor should act consistently as an investor and not as a speculator.", "author": "Ben Graham"},
                {"quote": "Price is what you pay. Value is what you get.", "author": "Warren Buffett"},
                {"quote": "In investing, what is comfortable is rarely profitable.", "author": "Robert Arnott"},
                {"quote": "The four most dangerous words in investing are: 'this time it's different.'", "author": "Sir John Templeton"},
                {"quote": "Risk comes from not knowing what you are doing.", "author": "Warren Buffett"},
                {"quote": "Wide diversification is only required when investors do not understand what they are doing.", "author": "Warren Buffett"}
            ])
        q = st.session_state.quote
        st.write(f'"{q["quote"]}"')
        st.caption(f"— {q['author']}")
        if st.button("Ganti Kutipan"):
            st.session_state.quote = random.choice([
                {"quote": "The stock market is filled with individuals who know the price of everything, but the value of nothing.", "author": "Philip Fisher"},
                {"quote": "Investing should be more like watching paint dry or watching grass grow.", "author": "Paul Samuelson"},
                {"quote": "The most important quality for an investor is temperament, not intellect.", "author": "Warren Buffett"},
                {"quote": "Know what you own, and know why you own it.", "author": "Peter Lynch"},
                {"quote": "The individual investor should act consistently as an investor and not as a speculator.", "author": "Ben Graham"},
                {"quote": "Price is what you pay. Value is what you get.", "author": "Warren Buffett"},
                {"quote": "In investing, what is comfortable is rarely profitable.", "author": "Robert Arnott"},
                {"quote": "The four most dangerous words in investing are: 'this time it's different.'", "author": "Sir John Templeton"},
                {"quote": "Risk comes from not knowing what you are doing.", "author": "Warren Buffett"},
                {"quote": "Wide diversification is only required when investors do not understand what they are doing.", "author": "Warren Buffett"}
            ])
            st.rerun()

# Footer
st.markdown("---")
st.caption("© 2025 Your Financial Companion | Michelle dan Prima Final Project")