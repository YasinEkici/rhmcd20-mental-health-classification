# Karar Ağacı ile RHMCD-20 Depresyon Tahmin Sistemi
YouTube videosu: https://youtu.be/vQqrv9JGUn8

Bu proje, RHMCD-20 veri setini kullanarak karar ağaçları (Decision Tree) ile depresyon tahmin sistemi geliştirmeyi amaçlamaktadır.

## Proje Hakkında

Bu çalışma, Bursa Teknik Üniversitesi Veri Madenciliğine Giriş dersi kapsamında Yasin Ekici tarafından hazırlanmıştır. Projede, RHMCD-20 veri seti kullanılarak bireylerin psikolojik durumu ve depresyon riskini tahmin etmek için bir karar ağacı modeli oluşturulmuştur. Model, özellikle "Coping_Struggles" (Başa Çıkma Zorlukları) hedef değişkenini ayırt etmede başarılı sonuçlar vermiştir.

## Veri Seti

Kullanılan veri seti "The RHMCD-20 datasets for Depression and Mental Health Data Analysis with Machine Learning" olarak adlandırılmıştır. İçeriği aşağıdaki gibidir:

* **Demografik Bilgiler:** Yaş aralığı (Age), Cinsiyet (Gender), Meslek (Occupation).
* **Karantina ve Psikososyal Faktörler:** Evde kalma süresi (Days_Indoors), Artan stres (Growing_Stress), Karantina frustrasyonları (Quarantine_Frustrations), Alışkanlık değişiklikleri (Changes_Habits).
* **Sağlık Geçmişi ve Fiziksel Belirtiler:** Daha önceki mental sağlık öyküsü (Mental_Health_History), Kilo değişimi (Weight_Change).
* **Psikolojik Bulgular:** Ruh hâli dalgalanmaları (Mood_Swings), Başa çıkma zorlukları (Coping_Struggles), İş ilgi düzeyi (Work_Interest), Sosyal zayıflık (Social_Weakness).

Veri setindeki birçok sütun "Yes/No/Maybe" veya kategorik aralıklarla (ör. "1-14 days," "More than 2 months") kodlanmıştır.

## Elde Edilen Sonuçlar

Projede Decision Tree modeli ile final hold-out testinde aşağıdaki sonuçlar elde edilmiştir:

* **Accuracy:** %61
* **ROC-AUC:** $\approx$0.68
* **PR-AUC:** $\approx$0.68

Model özellikle "Coping_Struggles" adlı hedef değişkeni ayırt etmede rastgele tahminden belirgin biçimde üstün çıkmıştır.

## Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, lütfen bir `pull request` açmaktan çekinmeyin veya bir `issue` oluşturarak önerilerinizi paylaşın.

## Lisans

Bu proje açık kaynaklıdır ve [Lisans Türü - örn: MIT Lisansı] altında lisanslanmıştır.

## Kaynakça

* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
* Scikit-learn'ün DecisionTreeClassifier ve DecisionTreeRegressor sınıflarını anlatan resmi dokümantasyon: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)
* [https://www.kaggle.com/learn/python](https://www.kaggle.com/learn/python)
* [https://www.kaggle.com/learn/intro-to-machine-learning](https://www.kaggle.com/learn/intro-to-machine-learning)
* [https://www.kaggle.com/learn/data-visualization](https://www.kaggle.com/learn/data-visualization)
