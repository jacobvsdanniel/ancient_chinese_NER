import os
import sys

import ner

def get_demo_data():
    data = {}
    data["person"] = [
        "N\t何岳鍾\t何能周悉。且伊既與岳鍾琪面談。\t琪所云三萬五千人進攻之說。竟全",
        "Y\t何岳鍾\t使司通政使。○以廣東澄海協副將\t、為陽江鎮總兵官。○以理藩院尚",
        "Y\t何岳鍾\t廣東陽江鎮總兵官。陽江鎮總兵官\t、為黃巖鎮總兵官。",
        "Y\t何岳鍾\t辦理防堵事務。浙江黃巖鎮總兵官\t、因病解任。以四川懋功協副將恆",
        "Y\t鄂賴\t等戶口、會造總冊投遞。因差郎中\t、遊擊黃喜林等、持銀牌、茶、緞",
        "Y\t鄂賴\t。應行令年羹堯等、俟差去之郎中\t等一到、將彼處情形、並所得一應",
        "Y\t鄂賴\t妄生疑慮。著一等侍衛達鼐。郎中\t。前往照管。俟將羅卜臧錫拉布等",
        "Y\t鄂賴\t都察院左副都御史○擢理藩院郎中\t、為內閣學士兼禮部侍郎。前往西",
        "Y\t鄂賴\t并奉先殿。得旨、是○命內閣學士\t、自藏至西寧。辦理蒙古事務○琉",
        "Y\t鄂賴\t拜、兼兵部侍郎行走○陞內閣學士\t、為理藩院額外侍郎",
    ]
    data["officer"] = [
        "Y\t總督閩浙\t日。新疆巡撫衙門八十二日。四川\t總督福建巡撫衙門四十八日。貴州",
        "N\t總督閩浙\t ○四十八年諭。兩江\t總督等合詞陳奏。以江浙兩省臣民",
        "Y\t總督閩浙\t皇帝南巡。先是四十八年諭。兩江\t總督等合詞陳奏。以江浙兩省臣民",
        "Y\t總督閩浙\t日。陝甘總督衙門四十一日。四川\t總督福建巡撫衙門四十八日。貴州",
        "Y\t總督閩浙\t浙撫。朝廷因材器使。原以左宗棠\t。軍務綦繁。 而浙省不可無人分任",
        "N\t總督閩浙\t議行。摺包○署福州將軍調補陝甘\t總督楊昌濬奏交卸督篆。將軍篆務",
        "N\t總督閩浙\t力保護。毋稍疏虞。摺包○署四川\t總督錫良奏、擬將川省腹地綠營額",
        "Y\t總督閩浙\t外任。荐膺疆寄。旋授尚書都統。\t。宣力有年。克勤厥職。茲以福建",
        "Y\t總督天津等處軍務\t太子太傅左都督駱養性、仍以原官\t。",
        "Y\t總督直隸山東河南\t ○十五年。裁舊設\t軍務一人、及標下中左右前後五營",
    ]
    data["location"] = [
        "N\t雙城\t學　墨爾根城義學　綏遠城義學　\t堡義學卷一千一百三十六八旗都統",
        "Y\t雙城\t廳學正、為府學教授。設伊通州、\t廳、各訓導一人。 ○又甘肅甯夏府",
        "Y\t雙城\t府。伯都訥、長春、五常、賓州、\t、五廳。伊通一州。 ○又新疆南路",
        "Y\t雙城\t、五常、三廳。撫民同知各一人。\t廳、撫民通判一人。吉林府、經歷",
        "Y\t雙城\t。五常廳、經歷一人。巡檢二人。\t廳、巡檢二人。伯都訥廳、巡檢二",
        "Y\t雙城\t各一人。改設府經歷一人。又添設\t廳撫民通判一人。巡檢二人。 ○又",
        "Y\t雙城\t補班次。悉照賓州廳五常廳之例。\t廳撫民通判。作為中缺。歸部銓選",
        "Y\t雙城\t由候補並揀發人員內。酌量咨補。\t廳、拉林、二巡檢。歸部銓選。如",
        "Y\t雙城\t蘭彩橋巡檢、儒學、攢典各一人。\t廳通判。典吏六人。司獄、巡檢、",
        "Y\t雙城\t。升吉林直隸廳為府。置伊通州、\t廳。隸府屬。 ○又改伯都訥理事同",
    ]
    data["organization"] = [
        "N\t宗　人　府\t部　郞　中　　臣汪　桂原　任　\t　主　事臣徐　煥內閣中書今",
        "Y\t山西道\t。皆於月之二十五日。兵部堂官會\t御史掣籤於　天安門外。若掣差官",
        "Y\t山西道\t議之件。而督以例限。每月於兵科\t註銷。",
        "Y\t山西道\t務府、奉宸苑、上駟院、武備院、\t御史、北城御史、鑲白旗、崇文門",
        "Y\t山西道\t。監察御史。滿洲一人。漢一人。\t掌印監察御史。滿洲一人。漢一人",
        "Y\t山西道\t十三倉。浙江道稽察禮部都察院。\t稽察兵部翰林院六科中書科總督倉",
        "Y\t山西道\t蘇安徽刑名。浙江道掌浙江刑名。\t掌山西刑名。山東道掌山東刑名。",
        "Y\t山西道\t。河南道御史監掣。武職月選籤。\t御史監掣。搭餉。局錢搭放兵餉。",
        "Y\t山西道\t畿道江南道各三人。河南道浙江道\t山東道陝西道湖廣道江西道福建道",
        "Y\t山西道\t西司廣西司雲南司。都察院河南道\t陝西道湖廣道江西道福建道廣西道",
    ]
    return data

def main():

    # Load model
    entity_type_to_model = {}
    for entity_type in ["person", "location", "organization", "officer"]:
        print(f"Loading {entity_type} model...")
    
        entity_type_to_model[entity_type] = ner.get_model(
            f"oldhan/{entity_type}.txt_train_character",
            "model",
            f"model_oldhan_{entity_type}.txt_BiLSTM-2-100_output-100-2_run1",
            hidden = "2-100",
            output = "100-2",
        )
        
    # Predict likelihood
    entity_type_to_line_list = get_demo_data()
    for entity_type in ["person", "officer", "location", "organization"]:
        print(f"Running {entity_type} model...")
        
        line_list = entity_type_to_line_list[entity_type]
        sample_list = [line.split("\t") for line in line_list]
        
        pl_list = ner.predict(
            entity_type_to_model[entity_type],
            sample_list,
            batch_samples = 256,
            batch_nodes = 8000,
        )
        
        for index, line in enumerate(line_list):
            print(f"{pl_list[index]:.2%}\t{line}")
    return
        
if __name__ == "__main__":
    main()
    sys.exit()
    
