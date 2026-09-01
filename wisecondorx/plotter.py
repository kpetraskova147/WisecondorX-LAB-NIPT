import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D

# --- Exact Color Scheme ---
COLOR_CHR_ODD = "#3A75A4"    # Steel Blue (chr1, chr3, ...)
COLOR_CHR_EVEN = "#D49B27"   # Amber Gold (chr2, chr4, ...)
COLOR_TRISOMY = "#E11D48"    # Crimson Red (Fetal Gain / +)
COLOR_MONOSOMY = "#14B8A6"   # Rich Teal (Fetal Loss / -)
COLOR_MATERNAL = "#555555"   # Charcoal Grey (Maternal 1n / 3n limits)
COLOR_BASELINE = "#000000"   # Solid Black (0.0 Center Baseline)
COLOR_GRID = "#E0E0E0"       # Faint Grid / Boundary Line
COLOR_GENE = "#9C4221"       # Gene Label Callouts


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
    # 1. Maternal Limits (Dotted Charcoal)
    ax.hlines(
        np.log2(1 / ploidy),
        start,
        end,
        colors=[COLOR_MATERNAL],
        linestyles="dotted",
        linewidth=1.5,
        zorder=4,
    )
    ax.hlines(
        np.log2(3 / ploidy),
        start,
        end,
        colors=[COLOR_MATERNAL],
        linestyles="dotted",
        linewidth=1.5,
        zorder=4,
    )

    # 2. Fetal Fraction Limits (Dashed Red / Teal)
    if beta is not None and ploidy > 0:
        loss_cutoff, gain_cutoff = get_aberration_cutoff(beta, ploidy)
        ax.hlines(
            gain_cutoff,
            start,
            end,
            colors=[COLOR_TRISOMY],
            linestyles="dashed",
            linewidth=1.5,
            zorder=4,
        )
        ax.hlines(
            loss_cutoff,
            start,
            end,
            colors=[COLOR_MONOSOMY],
            linestyles="dashed",
            linewidth=1.5,
            zorder=4,
        )

    # 3. Center Baseline (Solid Black, drawn on top of points)
    ax.hlines(
        np.log2(2 / ploidy),
        start,
        end,
        colors=[COLOR_BASELINE],
        linestyles="solid",
        linewidth=1.2,
        zorder=5,
    )


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
    zcutoff = float(data["zscore"])
    binsize = int(data["binsize"])
    n_reads_raw = int(data["n_reads"])
    n_reads_formatted = format_n_reads(n_reads_raw)
    ylim_arg = data["ylim"]
    regions_path = data["regions"]
    plot_title = data.get("plot_title")

    # --- Title Resolution: $(RUN_NUMBER)_$BARCODE or plot_title fallback ---
    run_number = os.environ.get("RUN_NUMBER", "").strip()
    barcode = os.environ.get("BARCODE", "").strip()

    if run_number and barcode:
        sample_title = f"{run_number}_{barcode}"
    elif run_number or barcode:
        sample_title = f"{run_number}{barcode}"
    else:
        sample_title = plot_title if plot_title else ""

    results_r = data["results_r"]
    results_w = data["results_w"]
    results_c = data["results_c"]

    # Flatten results_r and results_w for genome-wide plot
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

    # Short numeric labels for top scatter plot (1, 2, ..., 22, X, Y)
    short_labels = [l.replace("chr", "") for l in labels]

    # Assign alternating chromosome dot colors
    dot_cols = []
    for i in range(num_chrs):
        chr_color = COLOR_CHR_ODD if (i % 2 == 0) else COLOR_CHR_EVEN
        dot_cols.extend([chr_color] * bins_per_chr[i])

    # Boxplot data per chromosome
    box_list = []
    l_whis_per_chr = []
    h_whis_per_chr = []
    for i in range(num_chrs):
        chr_data = ratio[chr_ends[i] : chr_ends[i + 1]]
        box_list.append(chr_data[~np.isnan(chr_data)])
        stats = get_boxplot_stats(chr_data)
        l_whis_per_chr.append(stats[0])
        h_whis_per_chr.append(stats[1])

    chr_wide_upper_limit = 1.0
    chr_wide_lower_limit = -1.5

    if ylim_arg != "def":
        ylim_parts = re.findall(r"[-+]?\d*\.\d+|\d+", ylim_arg)
        if len(ylim_parts) == 2:
            chr_wide_lower_limit = float(ylim_parts[0])
            chr_wide_upper_limit = float(ylim_parts[1])

    # Parse target regions
    gene_labels = []
    if regions_path and os.path.exists(regions_path):
        with open(regions_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                chr_name = parts[0].replace("chr", "")
                if chr_name == "X":
                    c_idx = 22
                elif chr_name == "Y":
                    c_idx = 23
                else:
                    try:
                        c_idx = int(chr_name) - 1
                    except ValueError:
                        continue

                if c_idx >= num_chrs:
                    continue

                try:
                    r_start = int(parts[1])
                    r_end = int(parts[2])
                except ValueError:
                    continue

                start_bin = r_start // binsize + chr_ends[c_idx]
                end_bin = r_end // binsize + chr_ends[c_idx]

                if start_bin >= end_bin:
                    continue

                reg_ratio = np.nanmean(ratio[start_bin : end_bin + 1])
                label_pos = (
                    np.nanmax(ratio[start_bin : end_bin + 1]) + 0.2
                    if reg_ratio > 0
                    else np.nanmin(ratio[start_bin : end_bin + 1]) - 0.2
                )
                gene_labels.append(
                    {
                        "start_bin": start_bin,
                        "end_bin": end_bin,
                        "label": parts[3],
                        "pos": label_pos,
                        "adj": 0 if reg_ratio > 0 else 1,
                    }
                )

    # ---------------------------------------------------------
    # 1. GENOME-WIDE PLOT (2 rows, 3 columns layout)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.4, 1.0],
        width_ratios=[20, 1.2 if num_chrs > 22 else 0.01, 4.5],
        wspace=0.04,
        hspace=0.08
    )
    ax_main = fig.add_subplot(gs[0, :])
    ax_box_auto = fig.add_subplot(gs[1, 0])
    ax_box_sex = fig.add_subplot(gs[1, 1], sharey=ax_box_auto)
    ax_info = fig.add_subplot(gs[1, 2])

    # Top Scatter Plot
    ax_main.set_facecolor("white")
    ax_main.set_xlim(0, chr_ends[num_chrs])
    ax_main.set_ylim(chr_wide_lower_limit, chr_wide_upper_limit)
    ax_main.set_ylabel(r"$\log_2$ (ratio)", fontsize=11)

    # Reference lines (Maternal, Fetal, Baseline)
    if gender == "F":
        plot_threshold_lines(ax_main, 2, 0, chr_ends[num_chrs], beta=beta)
    else:
        plot_threshold_lines(ax_main, 2, 0, chr_ends[22], beta=beta)
        plot_threshold_lines(ax_main, 1, chr_ends[22], chr_ends[num_chrs], beta=beta)

    # Chromosome Boundary Separators
    for end in chr_ends[1:num_chrs]:
        ax_main.axvline(end, color=COLOR_GRID, linestyle="--", linewidth=0.8, zorder=1)

    # Masked Regions
    masked_indices = np.where(np.isnan(ratio))[0]
    for idx in masked_indices:
        ax_main.axvline(idx, color="#F3F4F6", linewidth=0.5, alpha=0.5, zorder=1)

    # Scatter points
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

    # Gene labels
    for gl in gene_labels:
        ax_main.scatter(
            range(gl["start_bin"], gl["end_bin"] + 1),
            ratio[gl["start_bin"] : gl["end_bin"] + 1],
            facecolors="none",
            edgecolors=COLOR_GENE,
            s=dot_sizes[gl["start_bin"] : gl["end_bin"] + 1] * 1.5,
            linewidths=1.5,
            zorder=3,
        )
        ax_main.text(
            (gl["start_bin"] + gl["end_bin"]) / 2,
            gl["pos"],
            gl["label"],
            color=COLOR_GENE,
            rotation=90,
            va="bottom" if gl["adj"] == 0 else "top",
            ha="center",
            fontsize=8,
            zorder=6,
        )

    # Aberration Segments
    for ab in results_c:
        chr_idx = int(ab[0])
        if chr_idx >= num_chrs:
            continue
        start = int(ab[1]) + chr_ends[chr_idx]
        end = int(ab[2]) + chr_ends[chr_idx]
        height = float(ab[4])
        color = COLOR_TRISOMY if height > 0 else COLOR_MONOSOMY
        face_color = list(plt.cm.colors.to_rgba(color))
        face_color[3] = 0.25
        ax_main.add_patch(
            plt.Rectangle(
                (start, 0),
                end - start,
                height,
                facecolor=face_color,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
        )
        ax_main.hlines(height, start, end, colors=[color], linewidth=2.5, zorder=5)

    # Chromosome numbers directly below top panel
    ax_main.set_xticks(chr_mids)
    ax_main.set_xticklabels(short_labels, fontsize=8, color="#333333")
    ax_main.tick_params(axis="x", length=3, color="#888888", pad=3)

    # Header title with sample identifier
#    if sample_title:
#        ax_main.set_title(sample_title, fontsize=13, fontweight="bold", pad=12, color="#1F2937")

    # ---------------------------------------------------------
    # 2. AUTOSOME BOXPLOTS (chr1 - chr22)
    # ---------------------------------------------------------
    auto_count = min(22, num_chrs)
    ax_box_auto.boxplot(
        box_list[:auto_count],
        labels=labels[:auto_count],
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="black", alpha=0.7),
        medianprops=dict(color="#D97706", linewidth=1.5),
        boxprops=dict(facecolor="white", color="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
    )
    ax_box_auto.set_ylim(-0.14, 0.14)
    ax_box_auto.set_ylabel(r"$\log_2$ (ratio)", fontsize=11)
    ax_box_auto.set_xticklabels(labels[:auto_count], rotation=45, ha="right", fontsize=9)
    ax_box_auto.grid(axis="y", alpha=0.3)

    plot_threshold_lines(ax_box_auto, 2, 0.5, auto_count + 0.5, beta=beta)

    # ---------------------------------------------------------
    # 3. SEX CHROMOSOME BOXPLOTS (chrX, chrY)
    # ---------------------------------------------------------
    if num_chrs > 22:
        ax_box_sex.boxplot(
            box_list[22:],
            labels=labels[22:],
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor="black", alpha=0.7),
            medianprops=dict(color="#D97706", linewidth=1.5),
            boxprops=dict(facecolor="white", color="black", linewidth=1),
            whiskerprops=dict(color="black", linewidth=1),
            capprops=dict(color="black", linewidth=1),
        )
        ax_box_sex.set_ylim(-0.14, 0.14)
        ax_box_sex.set_xticklabels(labels[22:], rotation=45, ha="right", fontsize=9)
        ax_box_sex.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
        ax_box_sex.grid(axis="y", alpha=0.3)

        sex_ploidy = 2 if gender == "F" else 1
        plot_threshold_lines(ax_box_sex, sex_ploidy, 0.5, len(labels[22:]) + 0.5, beta=beta)

    # ---------------------------------------------------------
    # 4. BOTTOM-RIGHT QC / METADATA INFO CARD
    # ---------------------------------------------------------
    ax_info.set_axis_off()
    bbox_props = dict(boxstyle="round,pad=0.8", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1)

    # NIPT Fetal Fraction Status
    ff_display = f"{beta * 100:.1f} %" if beta is not None else "N/A"

    # Sequencing QC and Accuracy Status
    seq_qc = os.environ.get("SEQ_QC_STATUS", "N/A")
    avg_id_raw = os.environ.get("AVG_IDENTITY", "").strip()
    accuracy_display = f"{avg_id_raw} %" if avg_id_raw else "N/A"

    # Safely format the sample title for Matplotlib's MathText (escaping underscores and spaces)
    display_title = sample_title if sample_title else "Sample Summary"
    safe_title = display_title.replace("_", r"\_").replace(" ", r"\ ")

    info_text = (
        f"$\\bf{{{safe_title}}}$\n\n"
        f"Reads: {n_reads_formatted}\n"
        f"Gender: {gender_label}\n"
        f"FF: {ff_display}\n"
        f"Accuracy: {accuracy_display}\n"
        f"Seq QC: {seq_qc}\n"
    )


    ax_info.text(
        0.05, 0.5, info_text,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="left",
        color="#1F2937",
        bbox=bbox_props,
        transform=ax_info.transAxes,
    )

    plt.savefig(os.path.join(out_dir, "genome_wide.png"), dpi=300, bbox_inches="tight")
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
        chr_dot_sizes = (chr_weights / np.pi) ** 0.5 * 30

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

        # Masked regions
        chr_masked = np.where(np.isnan(chr_ratio))[0]
        for idx in chr_masked:
            ax_chr.axvline(chr_start + idx, color="#F3F4F6", linewidth=1, alpha=0.5)

        # Scatter points
        ax_chr.scatter(
            range(chr_start, chr_end),
            chr_ratio,
            c=chr_dot_cols,
            s=chr_dot_sizes,
            edgecolors="none",
            alpha=0.8,
            zorder=2,
        )

        # Aberration Segments
        for ab in results_c:
            if int(ab[0]) != c:
                continue
            seg_start = int(ab[1]) + chr_start
            seg_end = int(ab[2]) + chr_start
            height = float(ab[4])
            color = COLOR_TRISOMY if height > 0 else COLOR_MONOSOMY
            face_color = list(plt.cm.colors.to_rgba(color))
            face_color[3] = 0.25
            ax_chr.add_patch(
                plt.Rectangle(
                    (seg_start, 0),
                    seg_end - seg_start,
                    height,
                    facecolor=face_color,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=3,
                )
            )
            ax_chr.hlines(height, seg_start, seg_end, colors=[color], linewidth=2.5, zorder=5)

        # Gene labels
        for gl in gene_labels:
            if gl["start_bin"] >= chr_start and gl["end_bin"] <= chr_end:
                ax_chr.scatter(
                    range(gl["start_bin"], gl["end_bin"] + 1),
                    ratio[gl["start_bin"] : gl["end_bin"] + 1],
                    facecolors="none",
                    edgecolors=COLOR_GENE,
                    s=chr_dot_sizes[gl["start_bin"] - chr_start : gl["end_bin"] - chr_start + 1] * 1.5,
                    linewidths=1.5,
                    zorder=3,
                )
                ax_chr.text(
                    (gl["start_bin"] + gl["end_bin"]) / 2,
                    gl["pos"],
                    gl["label"],
                    color=COLOR_GENE,
                    rotation=90,
                    va="bottom" if gl["adj"] == 0 else "top",
                    ha="center",
                    fontsize=9,
                    zorder=6,
                )

        # X-axis (MB)
        x_ticks = np.linspace(chr_start, chr_end, 11)
        x_labels = [f"{((t - chr_start) * binsize) / 1e6:.1f}" for t in x_ticks]
        ax_chr.set_xticks(x_ticks)
        ax_chr.set_xticklabels(x_labels)
        ax_chr.set_xlabel("Position (MB)", fontsize=10)

        plt.savefig(os.path.join(out_dir, f"{labels[c]}.png"), dpi=200, bbox_inches="tight")
        plt.close()
