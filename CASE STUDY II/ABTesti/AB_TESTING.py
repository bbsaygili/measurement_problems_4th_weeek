#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################
import pandas as pd

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi ve averagebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBidding uygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)



# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

dfc = pd.read_excel("/Users/bayramsaygili/Desktop/MIUUL DATA SCIENTIST 12TH TERM JUN-SEP23/4.WEEK/CASE STUDY II/ABTesti/ab_testing.xlsx", sheet_name="Control Group")
dft = pd.read_excel("/Users/bayramsaygili/Desktop/MIUUL DATA SCIENTIST 12TH TERM JUN-SEP23/4.WEEK/CASE STUDY II/ABTesti/ab_testing.xlsx", sheet_name="Test Group")



# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

dft.describe().T
dfc.describe().T


# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

# ayırt edici yeni bir sütun eklemek gerekiyor;

dfc["type"] = "control"
dft["type"] = "test"

#Grupların tek bir veride toplanması

concat_tc = pd.concat([dft, dfc], ignore_index=True)

concat_tc


#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

#  H0: mu1 = mu2, "Purchase"ler arasında fark yoktur
#  H1: mu1 != mu2,"Purchase"ler arasında fark vardır


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

dfc["Purchase"].mean()
dft["Purchase"].mean()



#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

#Önce şu aşağıdaki sıralamayı bi yazayım;

#1. Hipotezleri Kur (yukarıda kuruldu)
#2. Varsayım Kontrolü
#*  - 2.1. Normallik Varsayımı ( SHAPIRO WILK TESTİ:bir değişkenin dağılımının normal olup olmadığını test eder)
#*  - 2.2. Varyans Homojenliği(LEVENE: Bana iki farklı grup gönder,ben sana bu farklı iki gruba göre varyans homojenliğinin sağlanıp sağlanmadığını ifade edeyim..)
#3. Hipotezin Uygulanması
#   - 3.1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test) (T TEST, ttest_ind: Eğer normallik varsayamı sağlanıyorsa, beni kullan.NV ve VH sağlanıyosa,ok...NV sağlanıyor ama VH sağlanmıyorsa da beni kullan ama equal_var=False gir..Velch testi de yapmış oluruz.)
#   - 3.2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
#4. p-value değerine göre sonuçları yorumla
#Not:
#  - Normallik sağlanmıyorsa direkt 3.2 numara.Normallik sağlanıyor, Varyans homojenliği sağlanmıyorsa 3.1 numaraya gidilir ve şu arguman girilir; varyans homojenliği sağlanmadı.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.



######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

#NORMALLİK VARSAYIMI

#Control Grubu
test_stat, pvalue = shapiro(concat_tc.loc[concat_tc["type"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat= 0.9773, p-value = 0.5891 çıktı
# p-value = 0.5891 > 0.05 H0 REDDEDILEMEZ.

#Test Grubu
test_stat, pvalue = shapiro(concat_tc.loc[concat_tc["type"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = 0.9589, p-value = 0.1541
#p-value = 0.1541 > 0.05 H0 REDDEDILEMEZ.

#VARYANS HOMOJENLİĞİ

# H0: varyanslar homojendir
# H1: varyanslar homojen değildir

test_stat, pvalue = levene(concat_tc.loc[concat_tc["type"] == "control", "Purchase"],
                           concat_tc.loc[concat_tc["type"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Test Stat = 2.6393, p-value = 0.1083
#p-value = 0.1083 > 0.05 H0 REDDEDILEMEZ.



# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

#hem Normallik Varsayımı, hem de Varyans Homojenliği H0 Reddedilemedi !
#O zaman Parametrik Test'e !

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.


test_stat, pvalue = ttest_ind(concat_tc.loc[concat_tc["type"] == "control", "Purchase"],
                           concat_tc.loc[concat_tc["type"] == "test", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

#Test Stat = -0.9416, p-value = 0.3493
#p-value = 0.3493 > 0.05 H0 REDDEDILEMEZ.

#CONTROL GRUBU-TEST GRUBU ARASINDA ÖNEMLİ İSTATİSTİKİ BİR FARK YOKTUR,PARALAR BOŞA GİTTİ

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

#PARAMETRİK TEST => Çünkü, hem Normallik Varsayımı, Hem de Varyans Homojenliğinde H0 hipotezini
#rededemedik. O yüzden Parametrik Test ( Bağımsız iki örneklem T testi)


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# daha fazla kurcalamayın,bozarsınız