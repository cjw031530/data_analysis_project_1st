"""Export preprocessing comparisons as PNG, PDF, and a standalone HTML report."""

from __future__ import annotations

import argparse
import base64
import html
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from XGBoost.train import repo_path


def configure_plotting() -> None:
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    if "Malgun Gothic" in fonts:
        plt.rcParams["font.family"] = "Malgun Gothic"
    elif "NanumGothic" in fonts:
        plt.rcParams["font.family"] = "NanumGothic"
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})


def heatmap(ax, values, rows, columns, title):
    ax.imshow(values * 100, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title, loc="left", fontweight="bold", pad=14)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            position = (values[i, j] - values.min()) / max(float(np.ptp(values)), 1e-12)
            color = "white" if position > 0.65 else "#202936"
            ax.text(j, i, f"{values[i, j] * 100:.2f}%", ha="center", va="center", color=color, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_visualizations(output: Path) -> None:
    configure_plotting()
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    table = pd.read_csv(output / "comparison.csv")
    baseline = table.loc[table.phase.eq("screening")].reset_index(drop=True)
    methods = baseline.method.tolist()
    labels = baseline.label.tolist()
    test_months = list(report["protocol"]["test_weights"])
    weights = report["protocol"]["test_weights"]
    baseline_records = {record["method"]: record for record in report["results"] if record["phase"] == "screening"}
    test_values = np.array([[baseline_records[method]["test_by_month"][month]["misclassification_rate"]
                             for month in test_months] for method in methods])
    fold_values = np.array([[fold["metrics"]["misclassification_rate"]
                             for fold in baseline_records[method]["cv"]["folds"]] for method in methods])
    figure, axes = plt.subplots(2, 2, figsize=(17, 12), gridspec_kw={"width_ratios": [1.05, 1]})
    figure.suptitle("XGBoost 전처리 7가지 비교 · 공통 하이퍼파라미터", fontsize=20, fontweight="bold", y=0.99)
    current_score = float(baseline.loc[baseline.method.eq("current"), "test_weighted_score"].iloc[0])
    scores = baseline.test_weighted_score.to_numpy() * 100
    colors = ["#13795b" if score == min(scores) else "#2764a5" if method == "current" else "#a6b7ca"
              for method, score in zip(methods, scores)]
    ax = axes[0, 0]
    ax.barh(labels, scores, color=colors)
    ax.invert_yaxis()
    ax.axvline(current_score * 100, color="#2764a5", linestyle="--", linewidth=1.2, label="현재 전처리")
    zero_score = baseline_records["current"]["test"]["all_zero_weighted_score"] * 100
    ax.axvline(zero_score, color="#ba4a43", linestyle=":", linewidth=1.5, label="전부 0 예측")
    ax.set_xlim(0, max(zero_score, max(scores)) * 1.14)
    for i, score in enumerate(scores):
        ax.text(score + 0.04, i, f"{score:.3f}", va="center", fontweight="bold")
    ax.set_xlabel("가중 점수 ×100 · 낮을수록 좋음")
    ax.set_title("10~12번째 달 가중 오분류", loc="left", fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=9)
    heatmap(axes[0, 1], test_values, labels,
            [f"{month[:4]}-{month[4:]}\n가중치 {weights[month]}" for month in test_months],
            "테스트 월별 일반 오분류율")
    heatmap(axes[1, 0], fold_values, labels, ["F7 · 2023-12", "F8 · 2024-01", "F9 · 2024-02"],
            "Rolling 검증 월별 오분류율 · 선택에 사용")
    ax = axes[1, 1]
    counts = baseline.features.to_numpy()
    changes = (baseline.test_weighted_score.to_numpy() - current_score) * 100
    change_colors = ["#13795b" if value < 0 else "#bd544c" if value > 0 else "#2764a5" for value in changes]
    ax.barh([f"{label} (n={count:,})" for label, count in zip(labels, counts)], changes, color=change_colors)
    ax.invert_yaxis()
    margin = max(0.03, float(np.ptp(changes)) * 0.15)
    ax.set_xlim(min(0, changes.min()) - margin, max(0, changes.max()) + margin)
    ax.axvline(0, color="#2764a5", linewidth=1)
    for i, change in enumerate(changes):
        offset = margin * 0.15 * (-1 if change < 0 else 1)
        ax.text(change + offset, i, f"{change:+.3f}", va="center", ha="right" if change < 0 else "left")
    ax.set_title("현재 전처리 대비 차이 · n은 최종 입력 차원", loc="left", fontweight="bold", pad=14)
    ax.set_xlabel("가중 점수 차이 ×100 · 음수가 개선")
    figure.text(0.02, 0.015,
                "점수 = Σ(월별 가중치 × 오분류 여부) / 전체 테스트 행 수. 3월 0.1 · 4월 0.3 · 5월 0.6.\n"
                "각 방법의 임계값·반복 횟수는 F7~F9에서 결정. 범주 목록·결측 삭제는 학습 구간에서만 계산. 녹색 막대는 테스트 최저 점수.",
                fontsize=10, color="#48576a")
    figure.tight_layout(rect=(0, 0.06, 1, 0.965), h_pad=3, w_pad=3)
    save_figure(figure, output, "overview")

    tuned = table.loc[table.phase.eq("tuned")].reset_index(drop=True)
    trial_count = next(iter(report["tuning"].values()))["trials"] if report["tuning"] else 0
    if not tuned.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Rolling 검증 상위 {len(tuned)}개 · TPE 추가 효과", fontsize=17, fontweight="bold")
        positions = np.arange(len(tuned))
        for ax, field, title in zip(axes, ("cv_error", "test_weighted_score"),
                                    ("F7~F9 pooled 오분류율", "F10~F12 가중 점수 ×100")):
            before = np.array([baseline.loc[baseline.method.eq(method), field].iloc[0] for method in tuned.method]) * 100
            after = tuned[field].to_numpy() * 100
            ax.bar(positions - 0.18, before, width=0.36, color="#a6b7ca", label="공통 설정")
            ax.bar(positions + 0.18, after, width=0.36, color="#13795b", label=f"TPE {trial_count}회")
            ax.set_xticks(positions, tuned.label, rotation=8, ha="center")
            ax.set_title(title, loc="left", fontweight="bold")
            ax.set_ylim(0, max(0.1, max(before.max(), after.max()) * 1.18))
            for x, value in zip(positions - 0.18, before):
                ax.text(x, value + 0.05, f"{value:.3f}", ha="center", fontsize=10)
            for x, value in zip(positions + 0.18, after):
                ax.text(x, value + 0.05, f"{value:.3f}", ha="center", fontsize=10)
            ax.legend()
        fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=3)
        save_figure(fig, output, "tuning_comparison")

    rows = []
    tuned_label = f"TPE {trial_count}회"
    for _, row in table.iterrows():
        change = (row.test_weighted_score - current_score) * 100
        fields = [row.label, "공통 설정" if row.phase == "screening" else tuned_label,
                  f"{row.cv_error:.3%}", f"{row.test_weighted_score:.6f}", f"{change:+.3f}",
                  f"{row.test_error:.3%}", f"{row.test_auc:.4f}", f"{row.test_recall:.2%}", f"{int(row.features):,}"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in fields) + "</tr>")
    headings = ["방법", "설정", "Rolling error", "Weighted score ↓", "Δ vs current (×100)",
                "Test error", "Test AUC", "Test recall", "Features"]
    headers = "".join(f'<th onclick="sortTable({i})">{html.escape(value)} ↕</th>' for i, value in enumerate(headings))
    figures = []
    for name in ("overview", "tuning_comparison"):
        path = output / f"{name}.png"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            figures.append(f'<img alt="{name}" src="data:image/png;base64,{data}">')
    selected = report["selected_by_cv"]
    selected_label = next(r["label"] for r in report["results"] if r["method"] == selected["method"])
    document = """<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>XGBoost 전처리 비교</title>
<style>body{margin:0;background:#f2f5f8;color:#182739;font:16px/1.6 'Malgun Gothic',sans-serif}
main{max-width:1450px;margin:30px auto;padding:28px;background:white;border-radius:14px}
h1{margin-top:0}img{display:block;width:100%;height:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:11px;border-bottom:1px solid #dbe4ed;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{background:#eaf0f7;cursor:pointer}tr:hover{background:#f5f8fb}
.tablewrap{overflow:auto}.note{padding:16px;background:#eef4fb;border-left:4px solid #2764a5}
button{padding:9px 16px;margin:10px 5px 10px 0;border:1px solid #a6b7ca;background:white;border-radius:6px;cursor:pointer}
</style><main><h1>XGBoost 전처리 7가지 비교</h1>"""
    document += ('<p>Rolling validation: F7(2023-12), F8(2024-01), F9(2024-02). '
                 '최종 모델은 2024-02까지 학습하고 2024-03~05를 한 번에 평가했습니다.</p>')
    document += f'<p class="note">검증 기준 선택: <strong>{html.escape(selected_label)}</strong>. '
    document += ('테스트 점수는 Σ(0.1·3월 오분류 + 0.3·4월 오분류 + 0.6·5월 오분류) / 전체 행 수입니다. '
                 '일반 오분류율과 수치 크기가 다릅니다. 기존에 일부 평가 월을 살펴본 탐색적 비교입니다.</p>')
    best_baseline = baseline.loc[baseline.test_weighted_score.idxmin()]
    improvement = (current_score - best_baseline.test_weighted_score) / current_score * 100
    document += (f'<p><strong>공통 설정 테스트 최저: {html.escape(best_baseline.label)}</strong> · '
                 f'가중 점수 {best_baseline.test_weighted_score:.6f}, '
                 f'일반 오분류율 {best_baseline.test_error:.2%}. '
                 f'현재 전처리 {current_score:.6f} 대비 가중 점수가 {improvement:.2f}% 낮습니다. '
                 '이 순위는 최종 평가 결과이고, 위의 검증 기준 선택과 구분해서 읽어야 합니다.</p>')
    document += "".join(figures)
    document += ('<button onclick="filterRows(\'all\')">전체</button>'
                 '<button onclick="filterRows(\'공통 설정\')">7개 공통 설정</button>'
                 f'<button onclick="filterRows(\'{tuned_label}\')">TPE 후보</button>')
    document += f'<div class="tablewrap"><table id="results"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    document += """<p>CSV·JSON은 원본 숫자를 저장하며 PNG·PDF는 내보내기용입니다. 결측 50% 초과는 열 기준이고, 각 학습 데이터에서 계산합니다.
One-hot은 실제 수치형 0과 결측을 구분하는 dense float32 입력입니다. 원형도 문자열의 최소 숫자 변환이 필요합니다.</p>
<script>let directions={};function sortTable(c){const t=document.querySelector('#results tbody');const a=Array.from(t.rows);
directions[c]=-(directions[c]||-1);a.sort((x,y)=>{let p=x.cells[c].textContent,q=y.cells[c].textContent;
let u=Number(p.replace(/[%+,]/g,'')),v=Number(q.replace(/[%+,]/g,''));return directions[c]*(isNaN(u)||isNaN(v)?p.localeCompare(q):u-v)});a.forEach(r=>t.appendChild(r))}
function filterRows(p){document.querySelectorAll('#results tbody tr').forEach(r=>r.style.display=p==='all'||r.cells[1].textContent===p?'':'none')}
</script></main></html>"""
    (output / "report.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="XGBoost/outputs/preprocessing_experiment")
    args = parser.parse_args()
    create_visualizations(repo_path(args.output_dir))
