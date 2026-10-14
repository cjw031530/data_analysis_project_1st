"""Plot paired time-feature effects and export a standalone sortable report."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from XGBoost.calendar_preprocessing import BASE_METHODS
from XGBoost.train import repo_path
from XGBoost.visualize_preprocessing import configure_plotting, heatmap, save_figure


def create_visualizations(output: Path):
    configure_plotting()
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    table = pd.read_csv(output / "comparison.csv")
    pairs = pd.read_csv(output / "paired_comparison.csv")
    baseline = table.loc[table.phase.eq("screening")].reset_index(drop=True)
    records = {r["method"]: r for r in report["results"] if r["phase"] == "screening"}
    months = list(report["protocol"]["test_weights"])
    labels = [BASE_METHODS[m] for m in pairs.base_method]
    x = np.arange(len(pairs))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("XGBoost 전처리 3개 × 시간 포함 여부 · 공통 하이퍼파라미터", fontsize=19, fontweight="bold", y=0.99)
    ax = axes[0, 0]
    for shift, field, color, label in [(-0.18, "weighted_no_month", "#a6b7ca", "시간 제외"),
                                      (0.18, "weighted_with_month", "#2764a5", "시간 포함 (1~18)")]:
        values = pairs[field].to_numpy() * 100
        ax.bar(x + shift, values, width=0.36, color=color, label=label)
        for position, value in zip(x + shift, values):
            ax.text(position, value + 0.04, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(0.2, max(pairs.weighted_no_month.max(), pairs.weighted_with_month.max()) * 125))
    ax.set_ylabel("가중 점수 ×100 · 낮을수록 좋음")
    ax.set_title("F10~F12 평가 · 시간 제외/포함 비교", loc="left", fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax = axes[0, 1]
    for shift, field, color, label in [(-0.18, "cv_delta", "#9aa8b7", "Rolling 오분류율 변화"),
                                      (0.18, "weighted_delta", "#2764a5", "평가 가중 점수 변화")]:
        values = pairs[field].to_numpy() * 100
        ax.bar(x + shift, values, width=0.36, color=color, label=label)
        margin = max(0.008, max(abs(pairs.cv_delta).max(), abs(pairs.weighted_delta).max()) * 12)
        for position, value in zip(x + shift, values):
            ax.text(position, value + margin * (1 if value >= 0 else -1), f"{value:+.3f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
    ax.axhline(0, color="#48576a", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("시간 포함 - 제외 (×100) · 음수가 개선")
    ax.set_title("시간 특성의 추가 효과 · 서로 다른 점수 단위", loc="left", fontweight="bold", pad=12)
    delta_values = np.r_[pairs.cv_delta.to_numpy(), pairs.weighted_delta.to_numpy()] * 100
    delta_margin = max(0.04, float(np.ptp(delta_values)) * 0.35)
    ax.set_ylim(min(0, delta_values.min()) - delta_margin, max(0, delta_values.max()) + delta_margin)
    ax.legend(loc="best", fontsize=9)
    test_values = np.array([[records[m]["test_by_month"][month]["misclassification_rate"]
                             for month in months] for m in baseline.method])
    fold_values = np.array([[f["metrics"]["misclassification_rate"] for f in records[m]["cv"]["folds"]]
                           for m in baseline.method])
    heatmap(axes[1, 0], test_values, baseline.label.tolist(),
        [f"{m[:4]}-{m[4:]}\n가중치 {report['protocol']['test_weights'][m]}" for m in months], "평가 월별 일반 오분류율")
    heatmap(axes[1, 1], fold_values, baseline.label.tolist(),
            ["F7 · 2023-12", "F8 · 2024-01", "F9 · 2024-02"], "Rolling validation 월별 오분류율")
    fig.text(0.02, 0.018,
        "가중 점수 = Σ(월별 가중치 × 오분류 여부) / 전체 평가 행 수. 3월 0.1 · 4월 0.3 · 5월 0.6.\n"
        "임계값·반복 횟수는 F7~F9에서 결정. 범주형 상한 5의 고유값 수는 학습 구간에서만 계산. 시간은 202306=1, 202411=18.",
        fontsize=10, color="#48576a")
    fig.tight_layout(rect=(0, 0.065, 1, 0.965), h_pad=3.5, w_pad=3.5)
    save_figure(fig, output, "overview")

    tuned = table.loc[table.phase.eq("tuned")].reset_index(drop=True)
    trial_count = next(iter(report["tuning"].values()))["trials"] if report["tuning"] else 0
    tuned_label = f"TPE {trial_count}회"
    if not tuned.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(f"검증 상위 {len(tuned)}조건 · {tuned_label} 추가 효과", fontsize=17, fontweight="bold")
        x = np.arange(len(tuned))
        for ax, field, title in zip(axes, ("cv_error", "test_weighted_score"),
                                   ("Rolling 오분류율 (%)", "평가 가중 점수 ×100")):
            before = np.array([baseline.loc[baseline.method.eq(m), field].iloc[0] for m in tuned.method]) * 100
            after = tuned[field].to_numpy() * 100
            ax.bar(x - 0.18, before, width=0.36, color="#a6b7ca", label="공통 설정")
            ax.bar(x + 0.18, after, width=0.36, color="#13795b", label=tuned_label)
            ax.set_xticks(x, tuned.label.tolist(), fontsize=9)
            ax.set_ylim(0, max(0.2, max(before.max(), after.max()) * 1.18))
            ax.set_title(title, loc="left", fontweight="bold")
            for positions, values in [(x - 0.18, before), (x + 0.18, after)]:
                for position, value in zip(positions, values):
                    ax.text(position, value + 0.05, f"{value:.3f}", ha="center", fontsize=10)
            ax.legend()
        fig.tight_layout(rect=(0, 0.03, 1, 0.91), w_pad=3)
        save_figure(fig, output, "tuning_comparison")

    selected = report["selected_by_cv"]
    selected_row = table.loc[table.method.eq(selected["method"]) & table.phase.eq(selected["phase"])].iloc[0]
    test_best = baseline.loc[baseline.test_weighted_score.idxmin()]
    overall_best = table.loc[table.test_weighted_score.idxmin()]
    selected_phase_label = "공통 설정" if selected_row.phase == "screening" else tuned_label
    overall_phase_label = "공통 설정" if overall_best.phase == "screening" else tuned_label
    rows = []
    for _, row in table.iterrows():
        values = [row.label, "공통 설정" if row.phase == "screening" else tuned_label,
                  f"{row.cv_error:.3%}", f"{row.test_weighted_score:.6f}", f"{row.test_error:.3%}",
                  f"{row.test_auc:.4f}", f"{row.test_recall:.2%}", str(int(row.features)),
                  "기존 결과 재사용" if row.reused else "이번 학습"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    headers = ["조건", "설정", "Rolling error", "Weighted score ↓", "Test error", "Test AUC", "Test recall", "Features", "출처"]
    document = """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>XGBoost 시간·범주형 기준 비교</title><style>
body{margin:0;background:#f2f5f8;color:#182739;font:16px/1.6 'Malgun Gothic',sans-serif}
main{max-width:1500px;margin:30px auto;padding:28px;background:white;border-radius:14px}img{width:100%;height:auto;margin:20px 0}
.note{padding:16px;background:#eef4fb;border-left:4px solid #2764a5}.tablewrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:11px;border-bottom:1px solid #dbe4ed;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:#eaf0f7;cursor:pointer}tr:hover{background:#f5f8fb}button{padding:8px 15px;margin:8px 5px 8px 0;border:1px solid #a6b7ca;background:white;border-radius:6px;cursor:pointer}
</style><main><h1>XGBoost 전처리 3개 × 시간 포함 여부</h1>
<p>Rolling validation: 2023년 12월, 2024년 1·2월. 최종 모델은 2024년 2월까지 학습해 3~5월을 일괄 평가했습니다.</p>
<p class="note">시간 특성: 202306=1부터 202411=18까지의 숫자형 month_index. 범주형 상한 5는 기존 숫자 범주 중 학습 고유값이 5개를 넘는 열을 수치형으로 전환합니다.
문자 범주와 boolean은 유지합니다. 가중 점수는 Σ(월별 가중치 × 오분류 여부) / 전체 평가 행 수이며 일반 오분류율과 다릅니다.</p>"""
    document += (f"<p><strong>검증 기준 선택: {html.escape(selected_row.label)} · {selected_phase_label}</strong>, "
                 f"Rolling 오분류율 {selected_row.cv_error:.3%}, 평가 가중 점수 {selected_row.test_weighted_score:.6f}.</p>"
                 f"<p><strong>공통 설정 평가 최저: {html.escape(test_best.label)}</strong>, "
                 f"가중 점수 {test_best.test_weighted_score:.6f}, 일반 오분류율 {test_best.test_error:.3%}. "
                 "검증 선택과 평가 순위는 구분해서 읽어야 합니다.</p>")
    document += (f"<p><strong>전체 결과 평가 최저: {html.escape(overall_best.label)} · {overall_phase_label}</strong>, "
                 f"가중 점수 {overall_best.test_weighted_score:.6f}, 일반 오분류율 {overall_best.test_error:.3%}, "
                 f"AUC {overall_best.test_auc:.4f}.</p>")
    for name in ("overview", "tuning_comparison"):
        if (output / f"{name}.png").exists():
            encoded = base64.b64encode((output / f"{name}.png").read_bytes()).decode("ascii")
            document += f'<img alt="{name}" src="data:image/png;base64,{encoded}">'
    document += ('<button onclick="filterRows(\'all\')">전체</button>'
                 '<button onclick="filterRows(\'공통 설정\')">6조건 공통 설정</button>'
                 f'<button onclick="filterRows(\'{tuned_label}\')">TPE 후보</button>')
    document += '<div class="tablewrap"><table id="results"><thead><tr>' + "".join(
        f'<th onclick="sortTable({i})">{html.escape(v)} ↕</th>' for i, v in enumerate(headers)) + '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    document += """<p>동일한 데이터·설정·라이브러리 버전의 기존 결과만 재사용했습니다. 동일 조건의 기존 TPE 결과도 재사용할 수 있습니다.
한 seed와 과거에 살펴본 평가 월을 사용한 탐색적 비교입니다. 미제공된 2024년 6~11월의 실제 성능을 측정한 결과는 아닙니다.</p>
<script>let directions={};function sortTable(c){const t=document.querySelector('#results tbody');let a=Array.from(t.rows);directions[c]=-(directions[c]||-1);
a.sort((x,y)=>{let p=x.cells[c].textContent,q=y.cells[c].textContent,u=Number(p.replace(/[%+,]/g,'')),v=Number(q.replace(/[%+,]/g,''));
return directions[c]*(isNaN(u)||isNaN(v)?p.localeCompare(q):u-v)});a.forEach(r=>t.appendChild(r))}
function filterRows(p){document.querySelectorAll('#results tbody tr').forEach(r=>r.style.display=p==='all'||r.cells[1].textContent===p?'':'none')}
</script></main></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="XGBoost/outputs/calendar_experiment")
    create_visualizations(repo_path(parser.parse_args().output_dir))
