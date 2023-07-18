
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)


###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

dfp = pd.read_csv("/Users/bayramsaygili/Desktop/MIUUL DATA SCIENTIST 12TH TERM JUN-SEP23/4.WEEK/CASE STUDY I /Rating Product&SortingReviewsinAmazon/amazon_review.csv")
dfp.shape
dfp.columns
dfp.head()
dfp["overall"].mean()

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
dfp["overall"].mean()


###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

#Verisetindeki değerlendirme yapılan son tarih
dfp["reviewTime"].max()

#Puanlamanın yapıldığı tarihi gösteren değişkenin veri tipinin tarih olarak değiştirilmesi
dfp["reviewTime"] = pd.to_datetime(dfp["reviewTime"])

#Bugünün tarihini, analiz tarihi olarak belirlenmesi
today_date = pd.to_datetime("2014-12-09 0:0:0")

#Veri setine yeni bir değişken olarak, geçen zamanı eklemek
dfp["elapsed_days"] = (today_date - dfp["reviewTime"]).dt.days


#Son 30 günde verilen puanların ortalaması
less_30 = dfp.loc[dfp["elapsed_days"] <= 30, "overall"].mean()
print("Son 30 günde verilen puanların ortalaması:", less_30)

#30-90 gün uzaklıkta verilen puanların ortalaması
bt_30_90 = dfp.loc[(dfp["elapsed_days"] > 30) & (dfp["elapsed_days"] <= 90), "overall"].mean()
print("30-90 gün uzaklıkta verilen puanların ortalaması:", bt_30_90)

#90-180 gün uzaklıkta verilen puanların ortalaması
bt_90_180 = dfp.loc[(dfp["elapsed_days"] > 90) & (dfp["elapsed_days"] <= 180), "overall"].mean()
print("90-180 gün uzaklıkta verilen puanların ortalaması", bt_90_180)

#180 günden eski verilen puanların ortalaması
more_180 = dfp.loc[dfp["elapsed_days"] > 180, "overall"].mean()
print("180 günden eski verilen puanların ortalaması", more_180)

#Adım 3:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.

#ağırlıklı ortalama (ağırlıkları kendi isteğime göre,rasgele dağıttım)
print("Zaman tabanlı ağırlıklı ortalama:", less_30 * 0.40  + bt_30_90 * 0.30 + bt_90_180 * 0.20 + more_180 * 0.10)



###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################


dfp["helpful_no"] = dfp["total_vote"] - dfp["helpful_yes"]



# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.,
# Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.




###################################################
# Adım 2. score_average_rating,score_pos_neg_diff ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################


def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)

dfp["score_average_rating"] = dfp.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

dfp

#dfp["score_average_rating"].value_counts()


def score_pos_neg_diff(helpful_yes, helpful_no ):
    return helpful_yes - helpful_no

dfp["score_pos_neg_diff"] = dfp.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

dfp
dfp["score_pos_neg_diff"].value_counts()


def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):

    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * helpful_yes / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


dfp["wilson_lower_bound"] = dfp.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

dfp
dfp["wilson_lower_bound"].value_counts()


#as a summary;

#1
dfp.sort_values("wilson_lower_bound", ascending=False)
dfp.sort_values("score_pos_neg_diff", ascending=False)
dfp.sort_values("score_average_rating", ascending=False)

#2
by=["wilson_lower_bound", "score_pos_neg_diff", "score_average_rating"]
dfp.sort_values(by, ascending=[False, False, False])


#score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayabilmek için score_pos_neg_diff,
#score_average_rating ve wilson_lower_bound fonksiyonlarını tanımlayınız.
#score_pos_neg_diff'a göre skorlar oluşturunuz. Ardından; df içerisinde score_pos_neg_diff ismiyle kaydediniz.
#score_average_rating'a göre skorlar oluşturunuz. Ardından; df içerisinde score_average_rating ismiyle kaydediniz.
#wilson_lower_bound'a göre skorlar oluşturunuz. Ardından; df içerisinde wilson_lower_bound ismiyle kaydediniz.
ab_testing.py



##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################



# wilson_lower_bound'a göre ilk 20 yorumu belirleyip sıralayanız.
# Sonuçları yorumlayınız.

dfp[["wilson_lower_bound", "summary","reviewText" ]].head(20)