# 起點穩定度與相容賽局參照

正式來源：`outputs/contribution_start_stability/task11_formal_v1/analysis/`。固定20個模型團隊、兩個非零T，5反推×3求解起點，共600任務；全部有限box可行。模型、成員、均勻alpha固定；800次本地反推殘差均≤.1。這是既有模型的起點擴充，統計以每情境10隊為單位，不將600任務或owner當成獨立樣本。

## 符號、資格與付款

反推軸固定求解組合係數，換候選及其暖起點，隊內30個全配對；求解軸固定候選換起點，15個配對。符號tol1e−10，錨點正分資格另用tauA；無正分時reserve保留。下列是八個cohort×T×起點軸條件中，隊均三分類符號改變率的最小／最大值，不是信賴區間：

| 角色 | F符號改變率範圍 | Fminus符號改變率範圍 |
|---|---|---|
| A | 32.9–45.9% | 32.8–45.8% |
| M | 11.7–33.2% | 26.9–40.7% |
| J | 12.8–32.5% | 29.3–49.6% |
| AM | 11.2–34.2% | 34.9–46.9% |
| 完整LOO | 30.4–42.8% | 32.5–47.7% |

完整LOO本身也受有限求解程序與起點影響，不能將其當成不變的參與者屬性。完整正負直接翻轉、正分資格與分配TV另列，不混同符號tol與統計顯著性。

影子政策沿原100基本費＋500 A池＋500 M池，兩角色皆可靠負分才取消基本費。以下為含reserve的總付款向量TV，每隊先平均配對，再平均10隊：

| 情境 | T | 變動因子 | F平均TV | Fminus平均TV |
|---|---|---|---|---|
| clean K3 | −.5 | 反推 | .378328 | .505367 |
| clean K3 | −.5 | 求解 | .304533 | .465144 |
| clean K3 | +.5 | 反推 | .344562 | .498174 |
| clean K3 | +.5 | 求解 | .308533 | .424138 |
| controlled_poison K5 | −.5 | 反推 | .261944 | .442422 |
| controlled_poison K5 | −.5 | 求解 | .214596 | .355529 |
| controlled_poison K5 | +.5 | 反推 | .312680 | .476431 |
| controlled_poison K5 | +.5 | 求解 | .290200 | .406240 |

例：clean/T+.5/反推軸，F基本費資格翻轉率.115556，描述95%區間[.075556,.155556]；付款TV .344562 [.296692,.384631]。Fminus對應.415556 [.393333,.437778]及.498174 [.461097,.532544]。所有區間是200000次隊bootstrap、rng20261124的描述結果，未設定穩定通過門檻或多重推論。

`budget_clean_2026101611`、T−.5、ID2在15組配置中的影子總額為0–1100：r4/w0的A=−.002109、M=−.005830，基本費取消；r0/w1的A=.007707、M=.006159，得到100+500+500。tauA=1e−8，變動明顯大於數值容差。這是所抽起點與既有影子規則的實際敏感性，不是實際付款，也不把乾淨／污染不同K的差異當成污染因果效應。

## 完全匹配的40個Shapley賽局

每隊每T的r0w0枚舉全部子聯盟；固定完整模型F_N評分器、共同全隊起點，v(C)=F_N(start)−F_N(X_C)，空聯盟留在起點、v(empty)=0。移除模型及候選後以剩餘模型目標重新搜尋；full及LOO端點精確重用。共同起點保留full錨點資訊，評分仍使用被移除模型，是明定條件式賽局。Fminus不構成單一共同效用，未拼成Shapley向量。真值效用另作離線不同評分參照。

下表均為同F_N效用，n10；TV是正分normalize含reserve的診斷分配：

| 情境 | T | J–Shapley平均Spearman | LOO–Shapley平均Spearman | J平均TV | LOO平均TV |
|---|---|---|---|---|---|
| clean | −.5 | .15 | .70 | .544899 | .545782 |
| clean | +.5 | .30 | .50 | .426462 | .426188 |
| controlled_poison | −.5 | −.11 | .20 | .491190 | .398481 |
| controlled_poison | +.5 | −.37 | −.13 | .501142 | .804052 |

數值說明局部J、完整LOO及精確Shapley是不同分數，不能互換為同一付款資格。精確是賽局邊際值的計算性質，不是公平真值；完整區間、符號及utility sum gap見matched表。

## 獨立歷史600賽局

[historical](historical/)保留原875列中600個三人game的精確Shapley／LOO參照，非上述40game。原best_of（含舊SLSQP差分設定）、各子聯盟自身候選平均起點，與新角色隊不相容。以x0=0、T=.5的誤差.5作無合作基準，v(S)=.5−truth_error(S)、v(empty)=0。

先每seed平均120games，再用5seed描述bootstrap：Spearman .610 [.574167,.640833]、符號一致.616667 [.565000,.676111]、正分配置TV .476345 [.438395,.514296]。1800個Shapley均為正，LOO的58/600game全reserve。若只把非空聯盟基準增加δ，Shapley每人平移δ/3而LOO不變；此解析核對說明無合作基準會影響符號及分配。

## 精簡檔案、成本與驗收

- `team_role_stability.csv`／`team_payment_stability.csv`：800／160筆隊內配對彙整；`role_summary.csv`／`payment_summary.csv`：400／64個隊層級區間。
- `owner_payment_ranges.csv`：320個owner×target×judge的15配置範圍，保留基本費資格是否改變。
- `run_summary.csv`／`preparation_summary.csv`：600任務與200實際反推準備；同一準備共用3個求解起點，不重複加總。
- `matched_reference.csv`／`matched_summary.csv`：160個J/LOO同F與離線truth比較、64個隊區間。全部rank有效，仍保留有效隊數與pair數。
- `acceptance_summary.json`：23runtime／21輸入鎖；1996796個cache miss逐模型重算、3802644 cache hits、232554分數、1200影子付款、40完整賽局核對；40基準任務及全部原事件與任務9精確一致。另獨立重建1800隊內配對、8160隊指標、528區間、320金額range、40全排列Shapley。
- `historical/acceptance_summary.json`：875列／600完整game／4200子集查找、71原artifact鎖、公式／排列／效率／baseline平移通過；主代理另以三人閉式公式核對全部600game與8區間。

新增role cache misses1922997，matched game增量73799，合計1996796；observer等價控制額外242941。本地反推800次、forward400800；模型訓練為既有來源成本。每任務共享cache，以上不是各角色獨立冷快取成本。獨立驗收另有1997596次模型計算；統計及歷史Shapley不新增模型query或solver。smoke四任務額外7896 misses、3006反推forward及990 observer misses另列。全repo115 tests通過。

圖表由`python -m experiments.contribution.start_stability.plot --analysis RUN/analysis`產生，保留在忽略的outputs。研究查驗完成，現有付款規則的敏感性已量化；主分潤政策與使用條件留任務12決定。
