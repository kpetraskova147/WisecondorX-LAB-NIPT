import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

# --- Exact Color Scheme ---
COLOR_CHR_ODD = "#3A75A4"    
COLOR_CHR_EVEN = "#D49B27"   
COLOR_TRISOMY = "#E11D48"           # Fetal limit line for trisomy
COLOR_MONOSOMY = "#14B8A6"          # Fetal limit line for monosomy
COLOR_MATERNAL = "#520C61"          # Maternal limit line
COLOR_BASELINE = "#000000"          # Baseline of the upper graph
COLOR_GRID = "#999999"              # Boundary of chromosomes
COLOR_GENE = "#9C4221"              # I have no idea :D

def get_aberration_cutoff(beta, ploidy=2):
    loss_cutoff = np.log2((ploidy - (beta / 2)) / ploidy)
    gain_cutoff = np.log2((ploidy + (beta / 2)) / ploidy)
    return loss_cutoff, gain_cutoff

def format_n_reads(n_reads):
    s = str(int(n_reads))
    parts = []
    while s:
        parts.append(s[-3:])
        s = s[:-3]
    return ".".join(reversed(parts))

def plot_threshold_lines(ax, ploidy, start, end, beta=None):
    ax.hlines(np.log2(1 / ploidy), start, end, colors=[COLOR_MATERNAL], linestyles="dotted", linewidth=1.5, zorder=4)
    ax.hlines(np.log2(3 / ploidy), start, end, colors=[COLOR_MATERNAL], linestyles="dotted", linewidth=1.5, zorder=4)

    if beta is not None and ploidy > 0:
        loss_cutoff, gain_cutoff = get_aberration_cutoff(beta, ploidy)
        ax.hlines(gain_cutoff, start, end, colors=[COLOR_TRISOMY], linestyles="dashed", linewidth=1.5, zorder=4)
        ax.hlines(loss_cutoff, start, end, colors=[COLOR_MONOSOMY], linestyles="dashed", linewidth=1.5, zorder=4)

    ax.hlines(np.log2(2 / ploidy), start, end, colors=[COLOR_BASELINE], linestyles="solid", linewidth=1.2, zorder=3)

def get_boxplot_stats(data):
    if len(data) == 0:
        return [np.nan, np.nan]
    q1, q3 = np.nanpercentile(data, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    lower = max(lower, np.nanmin(data))
    upper = min(upper, np.nanmax(data))
    return [lower, upper]

def write_plots(data):
    plt.switch_backend("Agg")
    out_dir = data["out_dir"]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    raw_gender = str(data["ref_gender"]).upper()
    gender = "F" if ("F" in raw_gender or "XX" in raw_gender) else "M"
    gender_label = "XX (Female)" if gender == "F" else "XY (Male)"

    beta = float(data["beta"]) if data.get("beta") not in [None, "None", "nan", ""] else None
    binsize = int(data["binsize"])
    n_reads_raw = int(data["n_reads"])
    n_reads_formatted = format_n_reads(n_reads_raw)
    ylim_arg = data["ylim"]
    regions_path = data["regions"]
    plot_title = data.get("plot_title")

    run_number = os.environ.get("RUN_NUMBER", "").strip()
    barcode = os.environ.get("BARCODE", "").strip()
    analysis_type = os.environ.get("ANALYSIS_TYPE", "").strip()

    if run_number and barcode:
        sample_title = f"{run_number}_{barcode}"
        if analysis_type:
            sample_title += f"_{analysis_type}"
    else:
        sample_title = plot_title if plot_title else ""

    results_r = data["results_r"]
    results_w = data["results_w"]
    results_c = data["results_c"]

    ratio = []
    weights = []
    bins_per_chr = []
    for chr_r in results_r:
        ratio.extend(chr_r)
        bins_per_chr.append(len(chr_r))
    for chr_w in results_w:
        weights.extend(chr_w)

    ratio = np.array(ratio, dtype=float)
    ratio[ratio == 0] = np.nan
    weights = np.array(weights, dtype=float)
    weights[weights == 0] = np.nan

    num_chrs = 24 if gender == "M" else 23
    total_bins = sum(bins_per_chr[:num_chrs])
    ratio = ratio[:total_bins]
    weights = weights[:total_bins]

    chr_ends = [0] + np.cumsum(bins_per_chr).tolist()
    chr_mids = [chr_ends[i] + bins_per_chr[i] / 2 for i in range(num_chrs)]

    labels = [f"chr{i + 1}" for i in range(num_chrs)]
    if num_chrs >= 23:
        labels[22] = "chrX"
    if num_chrs >= 24:
        labels[23] = "chrY"

    short_labels = [l.replace("chr", "") for l in labels]

    dot_cols = []
    for i in range(num_chrs):
        chr_color = COLOR_CHR_ODD if (i % 2 == 0) else COLOR_CHR_EVEN
        dot_cols.extend([chr_color] * bins_per_chr[i])

    box_list = []
    for i in range(num_chrs):
        chr_data = ratio[chr_ends[i] : chr_ends[i + 1]]
        box_list.append(chr_data[~np.isnan(chr_data)])

    chr_wide_upper_limit = 1.0
    chr_wide_lower_limit = -1.5

    if ylim_arg != "def":
        ylim_parts = re.findall(r"[-+]?\d*\.\d+|\d+", ylim_arg)
        if len(ylim_parts) == 2:
            chr_wide_lower_limit = float(ylim_parts[0])
            chr_wide_upper_limit = float(ylim_parts[1])

    gene_labels = []

    # ---------------------------------------------------------
    # 1. GENOME-WIDE PLOT (Dynamic Layout Setup)
    # ---------------------------------------------------------
    legend_pos = os.environ.get("LEGEND_PLACEMENT", "bottom").strip().lower()

    fig = plt.figure(figsize=(13, 10))
    sex_width = num_chrs - 22 if num_chrs > 22 else 0.01

    if legend_pos == "right":
        gs = gridspec.GridSpec(
            2, 3, 
            height_ratios=[1.7, 1.0],
            width_ratios=[22, sex_width, 4.5], 
            wspace=0.04, 
            hspace=0.15
        )
        ax_main = fig.add_subplot(gs[0, :])
        ax_box_auto = fig.add_subplot(gs[1, 0])
        ax_box_sex = fig.add_subplot(gs[1, 1], sharey=ax_box_auto)
        ax_info = fig.add_subplot(gs[1, 2])
    else:
        # Default to Bottom
        gs = gridspec.GridSpec(
            3, 2, 
            height_ratios=[1.7, 1.0, 0.12], 
            width_ratios=[22, sex_width], 
            wspace=0.04, 
            hspace=0.15
        )
        ax_main = fig.add_subplot(gs[0, :])
        ax_box_auto = fig.add_subplot(gs[1, 0])
        ax_box_sex = fig.add_subplot(gs[1, 1], sharey=ax_box_auto)
        ax_info = fig.add_subplot(gs[2, :])

    ax_main.set_facecolor("white")
    ax_main.set_xlim(0, chr_ends[num_chrs])
    ax_main.set_ylim(chr_wide_lower_limit, chr_wide_upper_limit)
    ax_main.set_ylabel(r"$\log_2$ (ratio)", fontsize=11)

    if gender == "F":
        plot_threshold_lines(ax_main, 2, 0, chr_ends[num_chrs], beta=beta)
    else:
        plot_threshold_lines(ax_main, 2, 0, chr_ends[22], beta=beta)
        plot_threshold_lines(ax_main, 1, chr_ends[22], chr_ends[num_chrs], beta=beta)

    for end in chr_ends[1:num_chrs]:
        ax_main.axvline(end, color=COLOR_GRID, linestyle="--", linewidth=0.8, zorder=1)

    masked_indices = np.where(np.isnan(ratio))[0]
    for idx in masked_indices:
        ax_main.axvline(idx, color="#F3F4F6", linewidth=0.5, alpha=0.5, zorder=1)

    dot_sizes = (weights / np.pi) ** 0.5 * 20
    ax_main.scatter(
        range(len(ratio)),
        ratio,
        c=dot_cols,
        s=dot_sizes,
        edgecolors="none",
        alpha=0.8,
        zorder=2,
    )

    for ab in results_c:
        chr_idx = int(ab[0])
        if chr_idx >= num_chrs: continue
        start = int(ab[1]) + chr_ends[chr_idx]
        end = int(ab[2]) + chr_ends[chr_idx]
        height = float(ab[4])
        color = COLOR_TRISOMY if height > 0 else COLOR_MONOSOMY
        face_color = list(plt.cm.colors.to_rgba(color))
        face_color[3] = 0.25
        ax_main.add_patch(
            plt.Rectangle((start, 0), end - start, height, facecolor=face_color, edgecolor=color, linewidth=0.8, zorder=3)
        )
        ax_main.hlines(height, start, end, colors=[color], linewidth=2.5, zorder=5)

    ax_main.set_xticks(chr_mids)
    ax_main.set_xticklabels(short_labels, fontsize=8, color="#333333")
    ax_main.tick_params(axis="x", length=3, color="#888888", pad=2)

    # ---------------------------------------------------------
    # 2. AUTOSOME BOXPLOTS (chr1 - chr22)
    # ---------------------------------------------------------
    auto_count = min(22, num_chrs)
    ax_box_auto.boxplot(
        box_list[:auto_count],
        widths=0.4, # <-- ZÚŽENÍ KRABIČEK
        patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markersize=2.3, markerfacecolor="black", alpha=0.7),
        medianprops=dict(color="#D97706", linewidth=1.5, zorder=5),
        boxprops=dict(facecolor="white", color="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )
    ax_box_auto.set_ylim(-0.14, 0.14)
    ax_box_auto.set_xlim(0.5, auto_count + 0.5)
    ax_box_auto.set_ylabel(r"$\log_2$ (ratio)", fontsize=11)
    ax_box_auto.set_xticklabels(short_labels[:auto_count], rotation=0, ha="center", fontsize=9)
    ax_box_auto.grid(axis="y", alpha=0.3)
    plot_threshold_lines(ax_box_auto, 2, 0.5, auto_count + 0.5, beta=beta)

    # ---------------------------------------------------------
    # 3. SEX CHROMOSOME BOXPLOTS (chrX, chrY)
    # ---------------------------------------------------------
    if num_chrs > 22:
        ax_box_sex.boxplot(
            box_list[22:],
            widths=0.4, # <-- ZÚŽENÍ KRABIČEK
            patch_artist=True, showfliers=True,
            flierprops=dict(marker="o", markersize=2.3, markerfacecolor="black", alpha=0.7),
            medianprops=dict(color="#D97706", linewidth=1.5, zorder=5),
            boxprops=dict(facecolor="white", color="black", linewidth=1),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1),
        )
        ax_box_sex.set_ylim(-0.14, 0.14)
        ax_box_sex.set_xlim(0.5, len(labels[22:]) + 0.5)
        ax_box_sex.set_xticklabels(short_labels[22:], rotation=0, ha="center", fontsize=9)
        ax_box_sex.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
        ax_box_sex.grid(axis="y", alpha=0.3)
        sex_ploidy = 2 if gender == "F" else 1
        plot_threshold_lines(ax_box_sex, sex_ploidy, 0.5, len(labels[22:]) + 0.5, beta=beta)

    # ---------------------------------------------------------
    # 4. QC / METADATA INFO CARD
    # ---------------------------------------------------------
    ax_info.set_axis_off()
    
    bbox_props = dict(boxstyle="square,pad=0.8", facecolor="white", edgecolor="#888888", linewidth=1)

    ff_display = f"{beta * 100:.1f} %" if beta is not None else "N/A"
    seq_qc = os.environ.get("SEQ_QC_STATUS", "N/A")
    avg_id_raw = os.environ.get("AVG_IDENTITY", "").strip()
    accuracy_display = f"{avg_id_raw} %" if avg_id_raw else "N/A"
    display_title = sample_title if sample_title else "Sample Summary"
    safe_title = display_title.replace("_", r"\_").replace(" ", r"\ ")

    if legend_pos == "right":
        info_text = (
            f"$\\bf{{{safe_title}}}$\n\nReads: {n_reads_formatted}\nGender: {gender_label}\n"
            f"FF: {ff_display}\nAccuracy: {accuracy_display}\nSeq QC: {seq_qc}\n"
        )
        ax_info.text(0.45, 0.05, info_text, fontsize=10, va="bottom", ha="left", color="black", bbox=bbox_props, transform=ax_info.transAxes)
    else:
        info_text = (
            f"$\\bf{{{safe_title}}}$   |   Reads: {n_reads_formatted}   |   Gender: {gender_label}   |   "
            f"FF: {ff_display}   |   Accuracy: {accuracy_display}   |   Seq QC: {seq_qc}"
        )
        ax_info.text(0.5, 0.5, info_text, fontsize=11, va="center", ha="center", color="black", bbox=bbox_props, transform=ax_info.transAxes)

    plt.savefig(os.path.join(out_dir, "genome_wide.png"), dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()

    # ---------------------------------------------------------
    # 5. CHROMOSOME-SPECIFIC DETAIL PLOTS
    # ---------------------------------------------------------
    for c in range(num_chrs):
        fig_chr = plt.figure(figsize=(10, 6))
        ax_chr = fig_chr.add_subplot(111)

        chr_start = chr_ends[c]
        chr_end = chr_ends[c + 1]

        chr_ratio = ratio[chr_start:chr_end]
        chr_weights = weights[chr_start:chr_end]
        chr_dot_cols = dot_cols[chr_start:chr_end]
        chr_dot_sizes = (chr_weights / np.pi) ** 0.5 * 35

        stats = get_boxplot_stats(chr_ratio)
        if np.isnan(stats[0]):
            plt.close()
            continue

        upper_limit = 0.6 + stats[1]
        lower_limit = -1.05 + stats[0]
        upper_limit = max(upper_limit, np.nanmax(chr_ratio))
        lower_limit = min(lower_limit, np.nanmin(chr_ratio))

        if ylim_arg != "def":
            ylim_parts = re.findall(r"[-+]?\d*\.\d+|\d+", ylim_arg)
            if len(ylim_parts) == 2:
                lower_limit = float(ylim_parts[0])
                upper_limit = float(ylim_parts[1])

        ax_chr.set_xlim(chr_start, chr_end)
        ax_chr.set_ylim(lower_limit, upper_limit)

        chr_plot_title = f"{sample_title} - {labels[c]}" if sample_title else labels[c]
        ax_chr.set_title(chr_plot_title, fontsize=13, fontweight="bold", pad=10)
        ax_chr.set_ylabel(r"$\log_2$ (ratio)", fontsize=11)

        c_ploidy = 1 if (c >= 22 and gender == "M") else 2
        plot_threshold_lines(ax_chr, c_ploidy, chr_start, chr_end, beta=beta)

        chr_masked = np.where(np.isnan(chr_ratio))[0]
        for idx in chr_masked:
            ax_chr.axvline(chr_start + idx, color="#F3F4F6", linewidth=1, alpha=0.5)

        ax_chr.scatter(range(chr_start, chr_end), chr_ratio, c=chr_dot_cols, s=chr_dot_sizes, edgecolors="none", alpha=0.8, zorder=2)

        for ab in results_c:
            if int(ab[0]) != c: continue
            seg_start = int(ab[1]) + chr_start
            seg_end = int(ab[2]) + chr_start
            height = float(ab[4])
            color = COLOR_TRISOMY if height > 0 else COLOR_MONOSOMY
            face_color = list(plt.cm.colors.to_rgba(color))
            face_color[3] = 0.25
            ax_chr.add_patch(
                plt.Rectangle((seg_start, 0), seg_end - seg_start, height, facecolor=face_color, edgecolor=color, linewidth=0.8, zorder=3)
            )
            ax_chr.hlines(height, seg_start, seg_end, colors=[color], linewidth=2.5, zorder=5)

        x_ticks = np.linspace(chr_start, chr_end, 11)
        x_labels = [f"{((t - chr_start) * binsize) / 1e6:.1f}" for t in x_ticks]
        ax_chr.set_xticks(x_ticks)
        ax_chr.set_xticklabels(x_labels)
        ax_chr.set_xlabel("Position (MB)", fontsize=10)

        plt.savefig(os.path.join(out_dir, f"{labels[c]}.png"), dpi=200, bbox_inches="tight")
        plt.close()
