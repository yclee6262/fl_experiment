<callout icon="📌" color="blue_bg">
	**正式結論：尚未建立重抽樣品質初始化的穩定優勢。** 新的116個獨立seeds、三種情境、九項主要比較，校正區間全部跨零；低品質對B的普通95%區間小於零，但校正後仍跨零。保留A作一般基準、B作品質基準，重抽樣作研究對照。
</callout>
<table_of_contents/>

更新日期：<mention-date start="2026-10-08"/>。這份文件將舊結果、新探索與新正式結果分開。pooled evaluator 是事後主要品質判準；truth 是診斷。沒有依外部分數調整初始化、挑交付解、更新 α、pruning 或付款。

**狀態：** 探索、正式比較、重載及完整重播均已完成；研究未建立通用優勢，負面結果完整保留。

**主要結論：** 重抽樣在探索中讓α對盲測資料的波動降低約7%–10%，但新116-seed正式比較的九項校正區間全部跨零，尚未建立取代A/B或優於簡單收縮的穩定品質優勢。低品質對B的普通95%區間小於零，是改善訊號；經九比較校正後仍跨零。

**經濟判讀：** 資格預測重用避免新增標記和重複API查詢，但仍多本地重抽樣計算；相對B的Stage3查詢量只在乾淨情境略少，其他兩情境更多，不能宣稱普遍更省或金額收益。

已完成來源/模型/資料/權重/分數重載，以及seed4200三情境36組完整重播。保留A為一般基準、B為品質基準；重抽樣留作研究對照，暫不替換主要方法。

平均pooled誤差（target0，越小越好；每格116個有效seeds）：

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>A</td>
		<td>B</td>
		<td>簡單收縮</td>
		<td>重抽樣品質</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.017198</td>
		<td>0.016224</td>
		<td>0.015903</td>
		<td>0.015477</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.019092</td>
		<td>0.018125</td>
		<td>0.016526</td>
		<td>0.016572</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.018329</td>
		<td>0.016577</td>
		<td>0.016785</td>
		<td>0.016883</td>
	</tr>
</table>

## 研究問題與方法

研究問題：對小樣本盲測品質做重抽樣與平均，是否讓初始化比較穩定、改善交付 X，且值得新增計算成本？「α 比較平滑」不直接等於 X 較好。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>方法</td>
		<td>詳細定義</td>
		<td>需要的資訊</td>
	</tr>
	<tr>
		<td>A</td>
		<td>三人均為 1/3</td>
		<td>α 不依盲測品質變動；原資格檢查仍存在</td>
	</tr>
	<tr>
		<td>B</td>
		<td>α_i=(1/(MSE_i+1e-5))/Σ_j(1/(MSE_j+1e-5))</td>
		<td>同一組 n 筆盲測的逐人 MSE</td>
	</tr>
	<tr>
		<td>簡單收縮</td>
		<td>0.5×Uniform+0.5×B；比例固定</td>
		<td>與 B 相同</td>
	</tr>
	<tr>
		<td>重抽樣品質</td>
		<td>每次共同抽 n 行、有放回；每次算 B；平均1,000組正規化權重</td>
		<td>與 B 相同的逐筆誤差，可重用資格預測</td>
	</tr>
</table>

重抽樣是平均各次的「正規化權重」，不是先平均 MSE 再反比。模型不重訓，每個條件只做一次 Stage 3；best_of 內含兩條 solver 路徑，兩者查詢成本皆計入。α 是最佳化資訊權重，沒有把它當成報酬或公平分潤。

### 重抽樣品質：實際執行步驟與三人例子
1. 固定模型與參與者順序，取得原Stage 0的20筆盲測資料；對每個模型保存這20筆預測與逐筆平方誤差。
2. 從行號1–20有放回抽20個行號。同一輪所有模型用完全相同的行號，因此比較的是相同測試點；部分行號會重複，部分不會出現。
3. 分別計算各模型在這次抽樣的MSE，再以MSE加1e-5的倒數當分數，除以三人分數總和，得到一組非負、總和為1的權重。
4. 重複1,000次；將1,000組正規化權重逐分量平均，成為最後的初始α。不是先把1,000次MSE平均後再取倒數。
5. 使用這一組α執行一次Stage 3。內部best_of會比較兩條solver路徑，兩條的查詢成本都保存；不是執行1,000次倒推。
6. 交付X後，才使用凍結pooled evaluator評分；外部分數不回饋初始化、solver或選解。
**三人示意例，非實測資料：** 若三次抽樣得到權重(0.60,0.30,0.10)、(0.50,0.35,0.15)、(0.40,0.40,0.20)，平均即(0.50,0.35,0.15)。正式實驗平均的是1,000組，不是只用三組。重抽樣只重組現有20筆資訊，不會新增標記、重訓模型或證明誰應獲得較高付款。

## 做了哪些驗證

1. 10 個探索 seeds（4100–4109），每 seed 10 組獨立盲測；n=10／20／50／100；三情境、三 targets（-0.5、0、0.5）、四方法，共14,400條件。
2. 原 Stage0 20筆、target0 的120條件樣本數 pilot，決定新正式 N，不與正式結果合併。
3. 重抽樣次數100／500／1,000／10,000、五個 RNG 重複：先比較 α 數值誤差，再做原20筆、target0 的600次倒推，檢查 X 的敏感性。
4. 300 組實際 fresh／reuse 配對，核對 α、X、Stage3 查詢數一致。
5. 全部14,400條件重產生校準資料、重載模型／α／pooled分數、核對來源／資料／模型／結果雜湊與box。
6. 新正式 seeds4200–4315（116個），主要target0比較重抽樣對A／B／收縮，九項確認比較；另兩targets作探索性。

## 控制條件與範圍

固定二維、IDs [1,2,20]、三人、box[-1,1]、best_of、Custom30iterations、proposal500steps、原 Stage3 目標 Σα_i|f_i(X)-target|。乾淨／低品質／樣本量不平衡沿用前一輪設定，外部 evaluator 保持乾淨。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>coalition樣本數（ID1／ID2／ID20）</td>
		<td>local標記噪聲SD</td>
		<td>evaluator標記噪聲SD</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>500／500／500</td>
		<td>0.05／0.05／0.05</td>
		<td>全部0.05</td>
	</tr>
	<tr>
		<td>低品質25%</td>
		<td>500／500／500</td>
		<td>0.05／0.05／0.3</td>
		<td>全部0.05</td>
	</tr>
	<tr>
		<td>樣本量不平衡4×</td>
		<td>229／247／917</td>
		<td>全部0.05</td>
		<td>全部0.05</td>
	</tr>
</table>

低品質25%是全部20名訓練參與者中的最後5名增加噪聲；固定三人coalition只有ID20受影響。樣本量不平衡保持30epochs，但每epoch步數隨資料量變動，因此不能解讀成已控制每方相同訓練步數。BFGS命名路徑在box模式的實際solver為SLSQP，另一條為Custom；best_of依內部目標選可行解，不依pooled選解。

探索校準資料為獨立 Uniform([-1,1]^2)，精確合成標記與 Stage0 相同；不使用 pooled 訓練／驗證／測試資料。這不是宣稱真實部署可免費取得標記。探索固定已合格 coalition，不重新選人，因此測的是初始化對校準資料的敏感性。

四種 n 使用同一組100筆資料的前n筆，n比較為配對設計。每種n都用完全相同批次大小預測，保留逐筆資料與預測。seed 是統計單位，10次盲測平均後才跨seeds算區間。

## 探索：原本想降低的波動有沒有降低？

以下為 n=20、target0，各 seed 10 次盲測。α波動=三個權重分量的樣本變異數之和，越小表示對盲測資料越不敏感；pooled波動則是交付誤差的樣本變異，兩者不同。先在每seed算變異，再平均10seeds。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>方法</td>
		<td>α波動</td>
		<td>pooled誤差波動</td>
		<td>平均pooled誤差</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>A：Uniform</td>
		<td>0.000000</td>
		<td>0.00000000</td>
		<td>0.010635</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>B：品質</td>
		<td>0.017771</td>
		<td>0.00003231</td>
		<td>0.010532</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>簡單收縮</td>
		<td>0.004443</td>
		<td>0.00003603</td>
		<td>0.009881</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>重抽樣品質</td>
		<td>0.016061</td>
		<td>0.00002917</td>
		<td>0.009818</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>A：Uniform</td>
		<td>0.000000</td>
		<td>0.00000000</td>
		<td>0.020948</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>B：品質</td>
		<td>0.012959</td>
		<td>0.00001564</td>
		<td>0.010047</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>簡單收縮</td>
		<td>0.003240</td>
		<td>0.00002658</td>
		<td>0.009487</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>重抽樣品質</td>
		<td>0.011812</td>
		<td>0.00001607</td>
		<td>0.010153</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>A：Uniform</td>
		<td>0.000000</td>
		<td>0.00000000</td>
		<td>0.013278</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>B：品質</td>
		<td>0.024515</td>
		<td>0.00004599</td>
		<td>0.012351</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>簡單收縮</td>
		<td>0.006129</td>
		<td>0.00002293</td>
		<td>0.010849</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>重抽樣品質</td>
		<td>0.022867</td>
		<td>0.00004287</td>
		<td>0.011499</td>
	</tr>
</table>

重抽樣相對 B 的 α波動降低比例：

- 乾淨：9.6%。
- 低品質：8.9%。
- 樣本量不平衡：6.7%。

簡單收縮的 α變異為 B 的四分之一，是固定0.5混合公式直接導出的結果，不是發現它較有品質。它在某些情境的 pooled波動反而較大；因此權重較穩並不能單獨證明 X 較穩。

## 探索：盲測筆數與目標的影響

以下僅整理target0的平均pooled誤差；其他targets、每seed分布、CI、衝突及代表案例都保留在完整analysis.json。每格有10seeds×10次盲測，不當成100獨立seeds。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>n</td>
		<td>A</td>
		<td>B</td>
		<td>簡單收縮</td>
		<td>重抽樣</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>10</td>
		<td>0.010635</td>
		<td>0.010282</td>
		<td>0.010160</td>
		<td>0.009724</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>20</td>
		<td>0.010635</td>
		<td>0.010532</td>
		<td>0.009881</td>
		<td>0.009818</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>50</td>
		<td>0.010635</td>
		<td>0.009772</td>
		<td>0.010045</td>
		<td>0.009154</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>100</td>
		<td>0.010635</td>
		<td>0.009171</td>
		<td>0.009668</td>
		<td>0.009485</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>10</td>
		<td>0.020948</td>
		<td>0.011127</td>
		<td>0.008804</td>
		<td>0.011400</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>20</td>
		<td>0.020948</td>
		<td>0.010047</td>
		<td>0.009487</td>
		<td>0.010153</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>50</td>
		<td>0.020948</td>
		<td>0.010907</td>
		<td>0.008923</td>
		<td>0.010293</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>100</td>
		<td>0.020948</td>
		<td>0.010095</td>
		<td>0.009255</td>
		<td>0.010420</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>10</td>
		<td>0.013278</td>
		<td>0.013258</td>
		<td>0.010977</td>
		<td>0.012137</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>20</td>
		<td>0.013278</td>
		<td>0.012351</td>
		<td>0.010849</td>
		<td>0.011499</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>50</td>
		<td>0.013278</td>
		<td>0.011780</td>
		<td>0.010424</td>
		<td>0.011602</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>100</td>
		<td>0.013278</td>
		<td>0.011390</td>
		<td>0.010185</td>
		<td>0.011925</td>
	</tr>
</table>

以下三targets合計27項重抽樣對基準的探索比較（n=20），不作多重確認主張；每個區間以seed平均差估計。負差值表示重抽樣較好。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>target</td>
		<td>基準</td>
		<td>平均差</td>
		<td>95% CI</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>-0.5</td>
		<td>A：Uniform</td>
		<td>0.003812</td>
		<td>[-0.001296, 0.009884]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>-0.5</td>
		<td>B：品質</td>
		<td>0.000146</td>
		<td>[-0.000752, 0.001118]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>-0.5</td>
		<td>簡單收縮</td>
		<td>0.000990</td>
		<td>[-0.002131, 0.004002]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0</td>
		<td>A：Uniform</td>
		<td>-0.000816</td>
		<td>[-0.004326, 0.001638]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0</td>
		<td>B：品質</td>
		<td>-0.000713</td>
		<td>[-0.001492, 0.000059]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0</td>
		<td>簡單收縮</td>
		<td>-0.000063</td>
		<td>[-0.002788, 0.003063]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.5</td>
		<td>A：Uniform</td>
		<td>-0.000861</td>
		<td>[-0.012086, 0.008515]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.5</td>
		<td>B：品質</td>
		<td>-0.001014</td>
		<td>[-0.002204, -0.000050]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.5</td>
		<td>簡單收縮</td>
		<td>0.000175</td>
		<td>[-0.003104, 0.004359]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>-0.5</td>
		<td>A：Uniform</td>
		<td>-0.004489</td>
		<td>[-0.011011, 0.001477]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>-0.5</td>
		<td>B：品質</td>
		<td>-0.000597</td>
		<td>[-0.001645, 0.000266]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>-0.5</td>
		<td>簡單收縮</td>
		<td>-0.001261</td>
		<td>[-0.005326, 0.003101]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0</td>
		<td>A：Uniform</td>
		<td>-0.010795</td>
		<td>[-0.020454, -0.001511]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0</td>
		<td>B：品質</td>
		<td>0.000106</td>
		<td>[-0.000424, 0.000632]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0</td>
		<td>簡單收縮</td>
		<td>0.000666</td>
		<td>[-0.002033, 0.004200]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.5</td>
		<td>A：Uniform</td>
		<td>-0.009344</td>
		<td>[-0.025424, 0.002609]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.5</td>
		<td>B：品質</td>
		<td>-0.001186</td>
		<td>[-0.002297, -0.000295]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.5</td>
		<td>簡單收縮</td>
		<td>-0.003309</td>
		<td>[-0.008104, 0.000886]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>-0.5</td>
		<td>A：Uniform</td>
		<td>0.002157</td>
		<td>[-0.002306, 0.005493]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>-0.5</td>
		<td>B：品質</td>
		<td>-0.001032</td>
		<td>[-0.002611, 0.000385]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>-0.5</td>
		<td>簡單收縮</td>
		<td>0.001570</td>
		<td>[-0.000149, 0.003268]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0</td>
		<td>A：Uniform</td>
		<td>-0.001779</td>
		<td>[-0.007410, 0.002454]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0</td>
		<td>B：品質</td>
		<td>-0.000852</td>
		<td>[-0.001614, 0.000003]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0</td>
		<td>簡單收縮</td>
		<td>0.000650</td>
		<td>[-0.001199, 0.002373]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.5</td>
		<td>A：Uniform</td>
		<td>-0.001167</td>
		<td>[-0.006459, 0.004519]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.5</td>
		<td>B：品質</td>
		<td>-0.000125</td>
		<td>[-0.001427, 0.001246]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.5</td>
		<td>簡單收縮</td>
		<td>0.000740</td>
		<td>[-0.001863, 0.003193]</td>
	</tr>
</table>

## 重抽樣次數的數值穩定與交付品質

重抽樣次數增加通常讓權重更接近高次數參考，但不能從這點推論交付品質一定改善。以下為原 Stage0 20筆、target0，10seeds×5 RNG重複的描述性結果。B10000是數值參考，不是用 pooled挑出的最佳解。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>重抽樣次數</td>
		<td>平均pooled誤差</td>
		<td>對同RNG B10000的平均最大α分量差</td>
		<td>初始化秒數</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>100</td>
		<td>0.010532</td>
		<td>0.007477</td>
		<td>0.000346</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>500</td>
		<td>0.009749</td>
		<td>0.003211</td>
		<td>0.000502</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>1000</td>
		<td>0.010363</td>
		<td>0.002331</td>
		<td>0.000785</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>10000</td>
		<td>0.010845</td>
		<td>0.000000</td>
		<td>0.005855</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>100</td>
		<td>0.009454</td>
		<td>0.006618</td>
		<td>0.000475</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>500</td>
		<td>0.011680</td>
		<td>0.002750</td>
		<td>0.000542</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>1000</td>
		<td>0.011589</td>
		<td>0.001963</td>
		<td>0.000757</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>10000</td>
		<td>0.011528</td>
		<td>0.000000</td>
		<td>0.005462</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>100</td>
		<td>0.011912</td>
		<td>0.007716</td>
		<td>0.000285</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>500</td>
		<td>0.013116</td>
		<td>0.003724</td>
		<td>0.000496</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>1000</td>
		<td>0.013704</td>
		<td>0.002470</td>
		<td>0.000723</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>10000</td>
		<td>0.013847</td>
		<td>0.000000</td>
		<td>0.005205</td>
	</tr>
</table>

上述50次不是50獨立seeds，不以其平均差作新的正式優勢主張。平均 X分量差也保留於 MC analysis；倒推具有非線性與多個候選解，α差很小時 X仍可能明顯不同。正式方法仍固定B=1000，沒有因這些結果改選次數。

## 快取驗收發現與修正

第一版探索將100筆預測的前20筆當成20筆資格快取，與fresh20筆的矩陣運算有浮點差異。300配對全部α不同，最大分量差約8.97e-7，最大X分量差約1.235。這不是改善證據，舊版本因重用驗收未通過保留為診斷。

新版每n以同批次形狀算資格預測，再重用於初始化。300配對α與X精確一致，Stage3查詢數相同。三情境皆避免3次重複批次呼叫、60個輸入點；沒有新增標記。這是避免重複工作的成本改善，沒有替重抽樣建立品質優勢。

## 重抽樣不必然降低風險：教學例子

這是數學教學例，不是本輪實測結果。假設兩個模型在20筆上的平方誤差：A有19筆0.01、1筆1.00；B每筆都是0.04。全樣本MSE為A=0.0595、B=0.04，直接B品質權重約A=40.2%、B=59.8%。

每次有放回抽20筆時，約35.8%的重抽樣完全沒有抽到A的大誤差；那些抽樣會給A約80%權重。對所有可能抽樣的正規化權重取精確平均，A約50.4%、B約49.6%。重抽樣可能逆轉原MSE排序，而不是永遠替較不穩定的模型降權。這只說明初始化規則的行為，不直接推論X的品質，也不代表已測過攻擊或誘因相容。

完整概率計算存於alpha_weight_outputs/bootstrap_quality_rare_error_example_20261008.json。

## 成本與經濟意義

初始化wall time包含完整本地重抽樣，不只API耗時。Stage3、原始訓練／資格／proposal、外部評分分開記錄。正式原20筆預測是既有資格資訊的重用，不能再計一次完整資格成本。探索更換盲測資料需要新增標記，B、收縮、重抽樣各需n筆，A的α不需要；原資格檢查仍存在。

重抽樣不增加新的品質資訊，改善若成立來自對既有資料的處理。是否划算要同時看交付品質、失敗、輸入計價、API呼叫計價、本地時間與標記費用；本研究沒有指定貨幣價格，不能宣稱金額收益。沒有報價真實性、誘因相容、付款公平或惡意操弄保證。

## 正式樣本數與判讀規則

原20筆的10-seed pilot最大配對SD=0.0148394，依預先規則估得116seeds。計畫解析度Δ=0.005是合成研究尺度，不是經應用驗證的成功或等效門檻。常態近似與pilot SD不代表已保證80%power，若真實改善只有約0.001，這輪仍可能不夠精確。

正式target0的三情境×三基準共九項，普通95%與Bonferroni同時95%區間並列。要主張穩定品質優勢，校正區間必須全部小於零（逐情境逐基準判定）、有效配對完整且失敗率不較高；不能把部分情境改善說成通用方法。另target±0.5探索結果不加入確認主張。正式結果不回饋調參，負面研究仍保留。

## 新正式結果

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>方法</td>
		<td>n有效</td>
		<td>pooled平均</td>
		<td>中位數</td>
		<td>Q1</td>
		<td>Q3</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>A：Uniform</td>
		<td>116</td>
		<td>0.017198</td>
		<td>0.014700</td>
		<td>0.006843</td>
		<td>0.025895</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>B：品質</td>
		<td>116</td>
		<td>0.016224</td>
		<td>0.012479</td>
		<td>0.006720</td>
		<td>0.022738</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>簡單收縮</td>
		<td>116</td>
		<td>0.015903</td>
		<td>0.014436</td>
		<td>0.006151</td>
		<td>0.022880</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>重抽樣品質</td>
		<td>116</td>
		<td>0.015477</td>
		<td>0.011504</td>
		<td>0.005931</td>
		<td>0.020997</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>A：Uniform</td>
		<td>116</td>
		<td>0.019092</td>
		<td>0.013244</td>
		<td>0.007589</td>
		<td>0.028193</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>B：品質</td>
		<td>116</td>
		<td>0.018125</td>
		<td>0.015125</td>
		<td>0.008141</td>
		<td>0.023888</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>簡單收縮</td>
		<td>116</td>
		<td>0.016526</td>
		<td>0.014388</td>
		<td>0.004940</td>
		<td>0.024477</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>重抽樣品質</td>
		<td>116</td>
		<td>0.016572</td>
		<td>0.013294</td>
		<td>0.005915</td>
		<td>0.023650</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>A：Uniform</td>
		<td>116</td>
		<td>0.018329</td>
		<td>0.014639</td>
		<td>0.006581</td>
		<td>0.027521</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>B：品質</td>
		<td>116</td>
		<td>0.016577</td>
		<td>0.012762</td>
		<td>0.006170</td>
		<td>0.025205</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>簡單收縮</td>
		<td>116</td>
		<td>0.016785</td>
		<td>0.013657</td>
		<td>0.005136</td>
		<td>0.024558</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>重抽樣品質</td>
		<td>116</td>
		<td>0.016883</td>
		<td>0.012547</td>
		<td>0.004932</td>
		<td>0.024736</td>
	</tr>
</table>

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>基準</td>
		<td>重抽樣−基準</td>
		<td>95% CI</td>
		<td>校正區間</td>
		<td>穩定優勢</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>A：Uniform</td>
		<td>-0.001721</td>
		<td>[-0.003708, 0.000207]</td>
		<td>[-0.004484, 0.000993]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>B：品質</td>
		<td>-0.000747</td>
		<td>[-0.002292, 0.000671]</td>
		<td>[-0.003018, 0.001257]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>簡單收縮</td>
		<td>-0.000426</td>
		<td>[-0.001846, 0.000998]</td>
		<td>[-0.002409, 0.001573]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>A：Uniform</td>
		<td>-0.002520</td>
		<td>[-0.006023, 0.000833]</td>
		<td>[-0.007751, 0.002068]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>B：品質</td>
		<td>-0.001553</td>
		<td>[-0.003135, -0.000078]</td>
		<td>[-0.003836, 0.000379]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>簡單收縮</td>
		<td>0.000047</td>
		<td>[-0.002387, 0.002437]</td>
		<td>[-0.003385, 0.003317]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>A：Uniform</td>
		<td>-0.001446</td>
		<td>[-0.003806, 0.000824]</td>
		<td>[-0.004781, 0.001845]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>B：品質</td>
		<td>0.000305</td>
		<td>[-0.000724, 0.001365]</td>
		<td>[-0.001145, 0.001858]</td>
		<td>未建立</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>簡單收縮</td>
		<td>0.000098</td>
		<td>[-0.001929, 0.002147]</td>
		<td>[-0.002757, 0.002963]</td>
		<td>未建立</td>
	</tr>
</table>

## 正式成本、失敗與衝突

所有表格以target0為主。重用原Stage0預測後，各方法初始化新增API呼叫與新增標記皆為0；重抽樣仍有本地計算時間。原資格、模型訓練、proposal成本另外報告，不重複加入每個方法的初始化。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>方法</td>
		<td>有效／總數</td>
		<td>Stage3平均calls</td>
		<td>Stage3平均inputs</td>
		<td>初始化平均秒</td>
		<td>Stage3平均秒</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>A：Uniform</td>
		<td>116/116</td>
		<td>365.20</td>
		<td>365.20</td>
		<td>0.000024</td>
		<td>0.011413</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>B：品質</td>
		<td>116/116</td>
		<td>382.58</td>
		<td>382.58</td>
		<td>0.000054</td>
		<td>0.012000</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>簡單收縮</td>
		<td>116/116</td>
		<td>389.28</td>
		<td>389.28</td>
		<td>0.000054</td>
		<td>0.012167</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>重抽樣品質</td>
		<td>116/116</td>
		<td>375.54</td>
		<td>375.54</td>
		<td>0.000755</td>
		<td>0.011700</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>A：Uniform</td>
		<td>116/116</td>
		<td>379.86</td>
		<td>379.86</td>
		<td>0.000023</td>
		<td>0.011765</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>B：品質</td>
		<td>116/116</td>
		<td>365.72</td>
		<td>365.72</td>
		<td>0.000054</td>
		<td>0.011376</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>簡單收縮</td>
		<td>116/116</td>
		<td>370.53</td>
		<td>370.53</td>
		<td>0.000054</td>
		<td>0.011401</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>重抽樣品質</td>
		<td>116/116</td>
		<td>382.27</td>
		<td>382.27</td>
		<td>0.000775</td>
		<td>0.012793</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>A：Uniform</td>
		<td>116/116</td>
		<td>344.17</td>
		<td>344.17</td>
		<td>0.000023</td>
		<td>0.010685</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>B：品質</td>
		<td>116/116</td>
		<td>365.35</td>
		<td>365.35</td>
		<td>0.000054</td>
		<td>0.011296</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>簡單收縮</td>
		<td>116/116</td>
		<td>374.07</td>
		<td>374.07</td>
		<td>0.000054</td>
		<td>0.011555</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>重抽樣品質</td>
		<td>116/116</td>
		<td>374.12</td>
		<td>374.12</td>
		<td>0.000770</td>
		<td>0.011551</td>
	</tr>
</table>

模型／資格共用準備成本每seed保存一次；下表是正式來源整批共享的平均成本，沒有乘以四方法或三targets。evaluator訓練是研究評估成本，不是部署演算法成本。額外targets的proposal另在shared.jsonl保存。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>模型訓練秒</td>
		<td>資格秒</td>
		<td>原初proposal秒</td>
		<td>固定coalition proposal秒</td>
		<td>evaluator訓練秒</td>
		<td>資格calls／inputs</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>2.6340</td>
		<td>0.8916</td>
		<td>0.1361</td>
		<td>0.1343</td>
		<td>3.6307</td>
		<td>40.0／420.0</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>2.6454</td>
		<td>0.8889</td>
		<td>0.1329</td>
		<td>0.1332</td>
		<td>3.6399</td>
		<td>40.0／420.0</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>2.6427</td>
		<td>0.8929</td>
		<td>0.1339</td>
		<td>0.1385</td>
		<td>3.6329</td>
		<td>40.0／420.0</td>
	</tr>
</table>

正式來源runner為保存模型與資料，額外執行了每scenario/seed一次Uniform Stage3。這些準備用求解是研究流程的額外開銷，不算進任一方法單次部署的成本；其原始成本仍保存，整批研究成本不能漏掉。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>額外來源Uniform求解數</td>
		<td>平均Stage3 calls</td>
		<td>平均Stage3 inputs</td>
		<td>平均Stage3秒</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>116</td>
		<td>365.20</td>
		<td>365.20</td>
		<td>0.011341</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>116</td>
		<td>379.86</td>
		<td>379.86</td>
		<td>0.011694</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>116</td>
		<td>344.17</td>
		<td>344.17</td>
		<td>0.010787</td>
	</tr>
</table>

重載、雜湊核對、完整重播與其他驗收的本地運算屬研究驗證成本，沒有換算成外部API金額。這不是把未知成本填成零，亦不把驗證成本混進Stage3方法比較。

報告中的calls是本地模擬API呼叫，inputs是模型輸入點數；沒有假設每call或每point價格。查詢量減少不能在品質尚未證明相近時直接宣稱總體經濟優勢。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>重抽樣對基準</td>
		<td>pooled／truth方向衝突</td>
		<td>完整有效配對</td>
		<td>重抽樣／基準失敗</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>A：Uniform</td>
		<td>30</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>B：品質</td>
		<td>24</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>簡單收縮</td>
		<td>33</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>A：Uniform</td>
		<td>27</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>B：品質</td>
		<td>30</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>簡單收縮</td>
		<td>25</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>A：Uniform</td>
		<td>21</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>B：品質</td>
		<td>33</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>簡單收縮</td>
		<td>29</td>
		<td>116</td>
		<td>0／0</td>
	</tr>
</table>

衝突指同一seed、相同交付方法的品質差，在pooled與truth上的正負號相反；不表示模型檔案出錯，也不自動推翻主要評估。原始案例完整保留在analysis.json的conflicts。

## Evaluator覆蓋與適用限制

以下列出各情境target±0.25附近的測試樣本數與truth MSE，對每seed分別計算，再彙整。標記少於30樣本只是稀疏提示，不是經驗證的可靠性門檻，也不直接稱OOD。全域及hard-region評估由固定checkpoint保存，checkpoint僅以validation選擇。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>target</td>
		<td>附近樣本數：平均／最小</td>
		<td>稀疏seeds數</td>
		<td>附近truth MSE平均</td>
		<td>全域test truth MSE平均</td>
		<td>hard-region truth MSE平均</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>-0.5</td>
		<td>398.4／351</td>
		<td>0</td>
		<td>0.000113</td>
		<td>0.000137</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0</td>
		<td>262.7／230</td>
		<td>0</td>
		<td>0.000122</td>
		<td>0.000137</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.5</td>
		<td>184.3／150</td>
		<td>0</td>
		<td>0.000118</td>
		<td>0.000137</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>-0.5</td>
		<td>398.4／351</td>
		<td>0</td>
		<td>0.000113</td>
		<td>0.000137</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0</td>
		<td>262.7／230</td>
		<td>0</td>
		<td>0.000122</td>
		<td>0.000137</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.5</td>
		<td>184.3／150</td>
		<td>0</td>
		<td>0.000118</td>
		<td>0.000137</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>-0.5</td>
		<td>398.8／351</td>
		<td>0</td>
		<td>0.000119</td>
		<td>0.000136</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0</td>
		<td>264.1／227</td>
		<td>0</td>
		<td>0.000109</td>
		<td>0.000136</td>
		<td>0.000180</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.5</td>
		<td>185.0／155</td>
		<td>0</td>
		<td>0.000122</td>
		<td>0.000136</td>
		<td>0.000180</td>
	</tr>
</table>

## 不同target的獨立探索結果

以下target±0.5使用相同新的116seeds，但仍是探索性，不能加入九項主要確認比較。這裡的普通95%CI沒有因挑選有利target而轉成確認證據。

<table header-row="true" fit-page-width="true">
	<tr>
		<td>情境</td>
		<td>target</td>
		<td>基準</td>
		<td>重抽樣平均差</td>
		<td>95% CI</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>-0.5</td>
		<td>A：Uniform</td>
		<td>-0.000038</td>
		<td>[-0.002135, 0.002001]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>-0.5</td>
		<td>B：品質</td>
		<td>0.000065</td>
		<td>[-0.001225, 0.001413]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>-0.5</td>
		<td>簡單收縮</td>
		<td>0.001119</td>
		<td>[-0.000444, 0.002752]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.5</td>
		<td>A：Uniform</td>
		<td>-0.003667</td>
		<td>[-0.006472, -0.001203]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.5</td>
		<td>B：品質</td>
		<td>0.000247</td>
		<td>[-0.001102, 0.001498]</td>
	</tr>
	<tr>
		<td>乾淨</td>
		<td>0.5</td>
		<td>簡單收縮</td>
		<td>-0.001051</td>
		<td>[-0.003101, 0.000838]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>-0.5</td>
		<td>A：Uniform</td>
		<td>-0.000067</td>
		<td>[-0.002679, 0.002530]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>-0.5</td>
		<td>B：品質</td>
		<td>-0.000078</td>
		<td>[-0.002043, 0.001639]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>-0.5</td>
		<td>簡單收縮</td>
		<td>0.000683</td>
		<td>[-0.001915, 0.003247]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.5</td>
		<td>A：Uniform</td>
		<td>-0.004826</td>
		<td>[-0.007746, -0.001927]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.5</td>
		<td>B：品質</td>
		<td>0.000220</td>
		<td>[-0.000780, 0.001326]</td>
	</tr>
	<tr>
		<td>低品質</td>
		<td>0.5</td>
		<td>簡單收縮</td>
		<td>-0.001221</td>
		<td>[-0.003506, 0.001052]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>-0.5</td>
		<td>A：Uniform</td>
		<td>-0.001100</td>
		<td>[-0.003226, 0.000915]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>-0.5</td>
		<td>B：品質</td>
		<td>-0.000497</td>
		<td>[-0.001535, 0.000491]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>-0.5</td>
		<td>簡單收縮</td>
		<td>0.000544</td>
		<td>[-0.001091, 0.002212]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.5</td>
		<td>A：Uniform</td>
		<td>-0.003537</td>
		<td>[-0.006258, -0.000861]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.5</td>
		<td>B：品質</td>
		<td>0.000280</td>
		<td>[-0.001169, 0.001860]</td>
	</tr>
	<tr>
		<td>樣本量不平衡</td>
		<td>0.5</td>
		<td>簡單收縮</td>
		<td>-0.001362</td>
		<td>[-0.003830, 0.001053]</td>
	</tr>
</table>

## 完整重播與最後判讀

seed4200的三情境完整重播：通過；共36組方法/target對照。重生資料、切分、local/pooled模型、資格決策、proposal及解；最大數值差0。時間不要求相同。

九項主要比較中，0項建立校正後穩定優勢。

目前沒有足夠證據將重抽樣品質取代A/B，或將其視為優於簡單收縮的通用新初始化。這不證明等效或無效；其α較穩的機制效果可以保留，但品質主張需限縮。

主張邊界：本輪為合成二維、固定三人、原加權絕對誤差solver設定。未建立跨維度、不同coalition、真實資料、招募市場或付款公平的結論。已知數值求解對微小權重差敏感，應視為部署/重現限制，不能用事後pooled挑解掩蓋。

## 欄位字典與驗收

<table header-row="true" fit-page-width="true">
	<tr>
		<td>欄位</td>
		<td>意義</td>
	</tr>
	<tr>
		<td>scenario／seed／draw／blind_n／target／method</td>
		<td>完整唯一鍵；draw=-1為原Stage0盲測</td>
	</tr>
	<tr>
		<td>status</td>
		<td>valid、solver_failure、invalid_output、interrupted；不把程式錯誤混為研究失敗</td>
	</tr>
	<tr>
		<td>alpha／mse／alpha_audit</td>
		<td>最終初始化、該組盲測MSE、重抽樣分布資訊；不是付款</td>
	</tr>
	<tr>
		<td>X／max_box_violation</td>
		<td>交付解、最大超出box量；直接由X核對1e-8</td>
	</tr>
	<tr>
		<td>pooled_error／truth_error</td>
		<td>abs(預測-target)；前者主要、後者診斷；無交付留null</td>
	</tr>
	<tr>
		<td>initialization_seconds</td>
		<td>完整初始化計算wall time，包括bootstrap</td>
	</tr>
	<tr>
		<td>initialization_requests／input_samples</td>
		<td>初始化新增API呼叫／輸入筆數；重用版本應為0</td>
	</tr>
	<tr>
		<td>stage3_seconds／requests／input_samples</td>
		<td>兩條solver路徑合計成本，不含外部評分</td>
	</tr>
	<tr>
		<td>additional_labels</td>
		<td>額外校準標記；原盲測為0，探索B類為n</td>
	</tr>
	<tr>
		<td>calibration_*_independent</td>
		<td>某方法獨立取得校準預測的成本；不能重複加上已發生的資格費用</td>
	</tr>
	<tr>
		<td>shared.jsonl／calibration.jsonl</td>
		<td>共用proposal、coverage、校準資料及每n批次預測，整批只按實際共享成本計</td>
	</tr>
	<tr>
		<td>CI95／CI_simultaneous_9</td>
		<td>seed配對平均差的不確定性；不是單個解的誤差範圍</td>
	</tr>
	<tr>
		<td>conflict_run_pairs</td>
		<td>pooled與truth的品質差方向相反的案例數，保留識別鍵及差值</td>
	</tr>
</table>

探索14,400/14,400有效，無缺漏或重複；資料／模型預測／α／pooled重載最大差0，300重用配對通過。MC600/600有效、重載最大差0。回歸測試82項通過。正式與重播驗收不能由上述pilot代替。

## 檔案與重現

- 規格：docs/bootstrap_quality_research_execution_20261008.md。
- 探索登錄：docs/bootstrap_quality_mechanism_preregistration_v3_20261008.json。
- 正式登錄：docs/bootstrap_quality_independent_formal_preregistration_v2_20261008.json。
- 探索完整輸出：alpha_weight_outputs/bootstrap_quality_mechanism_v3_20261008。
- 原20筆樣本數pilot：alpha_weight_outputs/bootstrap_quality_power_pilot_20261008。
- 次數敏感性倒推：alpha_weight_outputs/bootstrap_quality_mc_delivery_20261008。
- 正式來源：alpha_weight_outputs/bootstrap_quality_independent_source_20261008。
- 正式結果：alpha_weight_outputs/bootstrap_quality_independent_formal_20261008。
- 執行管線：alpha_weight_outputs/bootstrap_quality_formal_pipeline_20261008/status.json；runner.py及分階段logs。

測試指令與全部有效參數／環境套件／hashes存於正式登錄。研究runner為bootstrap_quality_research.py，驗收與分析為analyze_bootstrap_quality_research.py。隔離新輸出執行，既有輸出不覆蓋。