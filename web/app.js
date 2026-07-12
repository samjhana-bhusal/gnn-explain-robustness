document.addEventListener("DOMContentLoaded", () => {
    let appData = null;
    let selectedNodeIdx = null;
    let activeExplainer = "none";
    let activeAttack = "random";
    let chartInstance = null;
    let simulation = null;

    console.log("hi..")
    // Load runs list catalog
    const runSelect = document.getElementById("run-select");

    fetch("runs/runs_list.json")
        .then(response => response.json())
        .then(catalog => {
            populateRuns(catalog.runs);
        })
        .catch(error => {
            console.warn("No runs catalog found, using default data.json", error);
            populateRuns([]);
        });

    function populateRuns(runs) {
        runSelect.innerHTML = "";
        
        // Add default run
        const defaultOpt = document.createElement("option");
        defaultOpt.value = "data.json";
        defaultOpt.textContent = "Active/Latest Run (data.json)";
        runSelect.appendChild(defaultOpt);
        
        // Add timestamped runs
        runs.forEach(run => {
            const opt = document.createElement("option");
            opt.value = run.path;
            opt.textContent = `${run.id} (${run.timestamp})`;
            runSelect.appendChild(opt);
        });
        
        // Default to data.json
        runSelect.value = "data.json";
        
        // Load initial data
        loadRunData(runSelect.value, true);
    }

    runSelect.addEventListener("change", (e) => {
        loadRunData(e.target.value, false);
    });

    function loadRunData(path, isInitial) {
        fetch(path)
            .then(response => response.json())
            .then(data => {
                appData = data;
                
                // Populate node dropdown
                const select = document.getElementById("node-select");
                select.innerHTML = "";
                appData.nodes.forEach(node => {
                    const option = document.createElement("option");
                    option.value = node.node_idx;
                    option.textContent = `Node #${node.node_idx} (Class ${node.true_class})`;
                    select.appendChild(option);
                });

                selectedNodeIdx = appData.nodes[0].node_idx;
                updateNodeInfo();
                renderGraph();

                if (isInitial) {
                    initChart();
                    attachEventListenersOnce();
                } else {
                    updateChart();
                }
            })
            .catch(error => {
                console.error("Error loading run data:", error);
                document.querySelector(".subtitle").innerText += ` (Error: Failed to load ${path})`;
            });
    }

    function attachEventListenersOnce() {
        const select = document.getElementById("node-select");
        
        // Listen for Node changes
        select.addEventListener("change", (e) => {
            selectedNodeIdx = parseInt(e.target.value);
            updateNodeInfo();
            renderGraph();
            updateChart();
        });

        // Listen for Attack strategy changes
        document.getElementById("attack-select").addEventListener("change", (e) => {
            activeAttack = e.target.value;
            updateChart();
        });

        // Listen for Explainer Highlights overlays
        const buttons = document.querySelectorAll(".explainer-selector-overlay button");
        buttons.forEach(button => {
            button.addEventListener("click", (e) => {
                buttons.forEach(btn => btn.classList.remove("active"));
                e.target.classList.add("active");
                activeExplainer = e.target.getAttribute("data-explainer");
                highlightExplanationEdges();
            });
        });
    }

    function getSelectedNodeData() {
        return appData.nodes.find(n => n.node_idx === selectedNodeIdx);
    }

    function updateNodeInfo() {
        const data = getSelectedNodeData();
        if (!data) return;

        // Statistics Bar
        document.getElementById("stat-true-class").innerText = data.true_class;
        document.getElementById("stat-pred-class").innerText = data.pred_class;
        document.getElementById("stat-confidence").innerText = `${(data.pred_prob * 100).toFixed(1)}%`;
        document.getElementById("stat-size").innerText = `${data.subgraph.nodes.length} nodes, ${data.subgraph.edges.length} edges`;

        // Fidelity Scoreboard
        // Helper to format fidelity values
        const formatFid = (val) => {
            const sign = val >= 0 ? "+" : "";
            return `${sign}${val.toFixed(3)}`;
        };

        const setFidelityCard = (prefix, metrics) => {
            const minusEl = document.getElementById(`${prefix}-fid-minus`);
            const plusEl = document.getElementById(`${prefix}-fid-plus`);
            minusEl.innerText = formatFid(metrics.fid_minus);
            plusEl.innerText = formatFid(metrics.fid_plus);
            
            // Highlight color based on ideal values
            // Fidelity minus is high drop (positive is good)
            minusEl.className = metrics.fid_minus > 0.15 ? "m-val highlight-green" : "m-val highlight-blue";
            // Fidelity plus is small drop (closer to 0 is good)
            plusEl.className = Math.abs(metrics.fid_plus) < 0.1 ? "m-val highlight-green" : "m-val highlight-purple";
        };

        setFidelityCard("gnn", data.fidelity.gnn_explainer);
        setFidelityCard("mcts", data.fidelity.subgraph_mcts);
        setFidelityCard("cf", data.fidelity.counterfactual);
    }

    // --- Chart.js Robustness Curve ---
    function initChart() {
        const ctx = document.getElementById("robustness-chart").getContext("2d");
        const nodeData = getSelectedNodeData();
        if (!nodeData) return;

        // Extract rates and values
        const rates = [1, 5, 10, 15]; // Percentages
        
        const chartData = getChartDatasets(nodeData);

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: rates.map(r => `${r}%`),
                datasets: chartData
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Inter', size: 11 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) label += context.parsed.y.toFixed(3);
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter' } },
                        title: { display: true, text: 'Topology Perturbation Rate', color: '#9ca3af', font: { family: 'Inter', size: 10 } }
                    },
                    y: {
                        min: 0,
                        max: 1.05,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter' } },
                        title: { display: true, text: 'Explanation Jaccard Similarity', color: '#9ca3af', font: { family: 'Inter', size: 10 } }
                    }
                }
            }
        });
    }

    function getChartDatasets(nodeData) {
        const extractJaccard = (explainerName) => {
            const list = nodeData.robustness[explainerName][activeAttack] || [];
            return list.map(item => item.jaccard);
        };

        return [
            {
                label: 'GNNExplainer',
                data: extractJaccard('gnn_explainer'),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2.5,
                tension: 0.15,
                fill: true
            },
            {
                label: 'Subgraph MCTS',
                data: extractJaccard('subgraph_mcts'),
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                borderWidth: 2.5,
                tension: 0.15,
                fill: true
            },
            {
                label: 'Counterfactual',
                data: extractJaccard('counterfactual'),
                borderColor: '#ec4899',
                backgroundColor: 'rgba(236, 72, 153, 0.1)',
                borderWidth: 2.5,
                tension: 0.15,
                fill: true
            }
        ];
    }

    function updateChart() {
        const nodeData = getSelectedNodeData();
        if (!nodeData || !chartInstance) return;

        const newDatasets = getChartDatasets(nodeData);
        chartInstance.data.datasets = newDatasets;
        chartInstance.update();
    }

    // --- D3.js Local Subgraph Visualization ---
    function renderGraph() {
        const nodeData = getSelectedNodeData();
        if (!nodeData) return;

        const svg = d3.select("#graph-svg");
        svg.selectAll("*").remove(); // Clear previous graph

        const width = document.querySelector(".graph-viewport-container").clientWidth;
        const height = document.querySelector(".graph-viewport-container").clientHeight;

        // Map class index to vibrant HSL color
        const colorScale = d3.scaleOrdinal()
            .domain([0, 1, 2, 3, 4, 5, 6])
            .range(['#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#f59e0b', '#06b6d4', '#a855f7']);

        // Deep copy nodes and links from dataset
        const nodes = nodeData.subgraph.nodes.map(d => Object.create(d));
        const links = nodeData.subgraph.edges.map(d => Object.create(d));

        // Create Force Simulation
        if (simulation) simulation.stop();
        simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(60))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(20));

        // Draw Links
        const link = svg.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("stroke", "rgba(255, 255, 255, 0.15)")
            .attr("stroke-width", 1.5)
            .attr("transition", "all 0.3s ease");

        // Tooltip setup
        const tooltip = d3.select("body").append("div")
            .attr("class", "graph-tooltip")
            .style("position", "absolute")
            .style("visibility", "hidden")
            .style("background", "rgba(17, 24, 39, 0.95)")
            .style("border", "1px solid rgba(255, 255, 255, 0.1)")
            .style("padding", "8px 12px")
            .style("border-radius", "6px")
            .style("font-size", "0.75rem")
            .style("color", "#fff")
            .style("pointer-events", "none")
            .style("z-index", "100")
            .style("backdrop-filter", "blur(4px)");

        // Draw Nodes
        const node = svg.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("r", d => d.is_target ? 12 : 7)
            .attr("fill", d => colorScale(d.label))
            .attr("stroke", d => d.is_target ? "#ff3e3e" : "rgba(255, 255, 255, 0.4)")
            .attr("stroke-width", d => d.is_target ? 3 : 1)
            .style("filter", d => d.is_target ? "drop-shadow(0px 0px 8px #ff3e3e)" : "none")
            .call(drag(simulation))
            .on("mouseover", (event, d) => {
                tooltip.style("visibility", "visible")
                    .html(`
                        <strong>Node ID:</strong> ${d.id}<br/>
                        <strong>True Label:</strong> Class ${d.label}<br/>
                        <strong>Pred Label:</strong> Class ${d.pred_label}<br/>
                        <strong>Confidence:</strong> ${(d.confidence * 100).toFixed(1)}%
                    `);
            })
            .on("mousemove", (event) => {
                tooltip.style("top", (event.pageY - 10) + "px")
                    .style("left", (event.pageX + 15) + "px");
            })
            .on("mouseout", () => {
                tooltip.style("visibility", "hidden");
            });

        // Add node labels for ID helper
        const labels = svg.append("g")
            .attr("class", "node-labels")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("dy", d => d.is_target ? -18 : -11)
            .attr("text-anchor", "middle")
            .attr("fill", "#9ca3af")
            .attr("font-size", "8px")
            .attr("font-family", "Inter")
            .text(d => d.is_target ? "TARGET" : `#${d.id}`);

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        });

        // Store active DOM handles for highlighting
        window.d3Links = link;
        window.d3Nodes = node;

        // Apply any current explanation highlights
        highlightExplanationEdges();
    }

    function highlightExplanationEdges() {
        const nodeData = getSelectedNodeData();
        if (!nodeData || !window.d3Links) return;

        const linkSelection = window.d3Links;
        
        if (activeExplainer === "none") {
            // Restore default lines
            linkSelection
                .attr("stroke", "rgba(255, 255, 255, 0.15)")
                .attr("stroke-width", 1.5)
                .style("filter", "none");
            return;
        }

        // Get the active explainer's target edges list
        const expEdges = nodeData.explanations[activeExplainer] || [];
        const expSet = new Set(expEdges.map(edge => `${edge[0]}-${edge[1]}`));

        // Set highlighting colors
        let highlightColor = "#3b82f6"; // default blue
        if (activeExplainer === "subgraph_mcts") highlightColor = "#8b5cf6"; // purple
        if (activeExplainer === "counterfactual") highlightColor = "#ec4899"; // pink

        linkSelection.each(function(d) {
            const u = d.source.id;
            const v = d.target.id;
            const key1 = `${u}-${v}`;
            const key2 = `${v}-${u}`;
            
            const isExplanatory = expSet.has(key1) || expSet.has(key2);
            
            if (isExplanatory) {
                d3.select(this)
                    .attr("stroke", highlightColor)
                    .attr("stroke-width", 3.5)
                    .style("filter", `drop-shadow(0px 0px 4px ${highlightColor})`);
            } else {
                d3.select(this)
                    .attr("stroke", "rgba(255, 255, 255, 0.05)")
                    .attr("stroke-width", 1.0)
                    .style("filter", "none");
            }
        });
    }

    // Drag behavior helper for D3
    function drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
        
        return d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended);
    }
});
