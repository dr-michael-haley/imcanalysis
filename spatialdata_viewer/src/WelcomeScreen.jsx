import React from "react";
import "./welcome.css";

export default function WelcomeScreen({
	recent,
	loading,
	onOpenFolder,
	onOpenZip,
}) {
	const directFolders = typeof window.showDirectoryPicker === "function";
	return (
		<section
			className="welcome-screen"
			aria-labelledby="welcome-title"
			tabIndex={0}
		>
			<div className="welcome-content">
				<header className="welcome-hero">
					<img
						className="welcome-logo"
						src="./logo.svg"
						alt="SpatialBiologyToolkit"
					/>
					<div>
						<h1 id="welcome-title">Welcome to MERSCOPE Explorer</h1>
						<p>
							Explore transcripts, compare cell segmentations and view gene
							expression in tissue — directly in your browser.
						</p>
					</div>
				</header>
				<div className="welcome-actions">
					<button className="primary" onClick={onOpenFolder} disabled={loading}>
						{directFolders
							? "Open local / network folder"
							: "Open SpatialData folder"}
					</button>
					<button onClick={onOpenZip} disabled={loading}>
						Open a ZIP
					</button>
				</div>
				<nav
					className="welcome-resources"
					aria-label="MERSCOPE Explorer resources"
				>
					<a
						href="https://imcanalysis.readthedocs.io/en/latest/browser_tools/merscope_explorer.html"
						target="_blank"
						rel="noopener noreferrer"
						title="Opens in a new tab"
					>
						MERSCOPE Explorer user guide
					</a>
					<a href="mailto:michael.haley@manchester.ac.uk?subject=SBT%20MERSCOPE%20Explorer">
						Contact Michael Haley
					</a>
				</nav>
				{recent.length > 0 && (
					<p className="local-resume">
						<strong>Saved workspace: {recent[0].name}</strong>
						<br />
						Reopen the same folder or ZIP using the same opening method to
						resume.
					</p>
				)}
				<div className="welcome-columns">
					<section
						className="welcome-card"
						aria-labelledby="welcome-features-title"
					>
						<h2 id="welcome-features-title">Explore your results</h2>
						<dl>
							<div>
								<dt>Find transcripts in tissue</dt>
								<dd>
									Show individual genes or a whole gene list as Points, with
									colours for each gene.
								</dd>
							</div>
							<div>
								<dt>Compare cell segmentations</dt>
								<dd>
									Link cell boundaries to their matching tables. Compare
									segmentation approaches in up to four linked views.
								</dd>
							</div>
							<div>
								<dt>Explore expression and annotations</dt>
								<dd>
									Colour cells by a gene, the summed expression of a gene list,
									or a cell annotation. Adjust colour maps and expression
									limits.
								</dd>
							</div>
							<div>
								<dt>Connect tissue to embeddings</dt>
								<dd>
									Explore existing UMAPs and other embeddings, with cell
									selections linked to the tissue view.
								</dd>
							</div>
							<div>
								<dt>Pick up where you left off</dt>
								<dd>
									Save views and gene lists in your workspace, import gene lists
									from CSV, and bookmark regions with comments.
								</dd>
							</div>
							<div>
								<dt>Export figures</dt>
								<dd>
									Arrange panes with legends, layer names and calibrated scale
									bars. Export PNG, JPEG or layered SVG.
								</dd>
							</div>
						</dl>
					</section>
					<section
						className="welcome-card"
						aria-labelledby="welcome-assets-title"
					>
						<h2 id="welcome-assets-title">Bring a SpatialData dataset</h2>
						<p className="welcome-card-intro">
							Open a SpatialData <code>.zarr</code> folder or ZIP containing the
							layers you want to explore.
						</p>
						<dl className="welcome-assets">
							<div>
								<dt>
									Points <span>transcripts</span>
								</dt>
								<dd>
									Transcript locations and gene identities, ready to filter by
									selected genes or a workspace gene list.
								</dd>
							</div>
							<div>
								<dt>
									Shapes <span>cell boundaries</span>
								</dt>
								<dd>
									Cell outlines for each segmentation approach. Add the
									boundaries you want to compare as active layers.
								</dd>
							</div>
							<div>
								<dt>
									Tables <span>cell measurements</span>
								</dt>
								<dd>
									AnnData tables stored inside SpatialData, with gene expression
									in <code>X</code> and cell annotations in <code>obs</code>.
									Cell IDs must match the linked boundaries.
								</dd>
							</div>
							<div>
								<dt>
									Images &amp; labels <span>tissue context</span>
								</dt>
								<dd>
									Multichannel images and segmentation masks, aligned using the
									coordinate systems stored in your dataset.
								</dd>
							</div>
						</dl>
						<details className="welcome-details">
							<summary>Opening files &amp; resuming a workspace</summary>
							<p>
								In desktop Chrome or Edge, use{" "}
								<strong>Open local / network folder</strong> for large datasets
								and mapped network drives. The compatibility folder picker is
								available in Setup.
							</p>
							<p>
								Workspaces remember your settings, not copies of your data.
								Reopen the same dataset using the same opening method to restore
								local settings. Export workspace JSON to move your settings to
								another browser or computer.
							</p>
						</details>
					</section>
				</div>
				<p className="welcome-local">
					<strong>Your files stay local.</strong> Dataset files are opened
					read-only, including files on accessible network drives. Nothing is
					uploaded and your source data is unchanged.
				</p>
			</div>
		</section>
	);
}
