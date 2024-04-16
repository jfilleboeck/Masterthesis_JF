// Creates and plots initial data on the graph
const plotData = [{
    x: xData,
    y: yData,
    mode: 'lines',
    type: 'scatter',
}];

const layout = {
    dragmode: 'select',
    autosize: true,
    responsive: true,
};

// Global variables for feature manipulation and display
let displayedFeature = document.getElementById('display-feature').value;
var valDataID = null;


// Initializes the plot with data and setups event listeners for interactive features
Plotly.newPlot('plot', plotData, layout).then(() => {
    console.log("Plot created");
    store_x_values();
    createHistogramPlot(hist_data, bin_edges);
});

// Creates a histogram plot with provided data
function createHistogramPlot(hist_data, bin_edges) {
    // Prepare data for the plot
    var trace = {
        x: bin_edges,
        y: hist_data.map(x => Math.abs(x)),
        type: 'bar',
        marker: {
            color: 'blue'
        },
        hoverinfo: 'x+y',
    };

    var layout = {
        xaxis: {
            title: 'Bins'
        },
        yaxis: {
            title: 'Frequency',
            tickmode: 'auto',
            nticks: 2
        },
        bargap: 0.05,
        height: 250
    };

    Plotly.newPlot('histogram-plot', [trace], layout);
}


// Sets up listeners for feature selection and initializes UI components
document.addEventListener('DOMContentLoaded', function() {

    const selectBox = document.getElementById('display-feature');
    // Event listener for feature selection change
    selectBox.addEventListener('change', function () {
        displayedFeature = selectBox.value;
        fetchFeatureData(displayedFeature);
    });
    predictAndGetMetrics();
    fetchDataAndCreateTable();

    const validationDataButton = document.getElementById('validation-data-button');
    const instanceExplanationsButton = document.getElementById('instance-explanations-button');
    const shapeFunctionsButton = document.getElementById('shape-functions-button');
    const correlationMatrixButton = document.getElementById('correlation-matrix-button');

    validationDataButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('datagrid-table').style.display = 'block';
    });

    instanceExplanationsButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('instance-explanations-content').style.display = 'block';
        fetchAndDisplayInstanceExplanation();
    });

    shapeFunctionsButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('shape-functions-content').style.display = 'block';
        fetchAndDisplayShapeFunctions();
    });

    correlationMatrixButton.addEventListener('click', function() {
        hideAllContentSections();
        document.getElementById('correlation-matrix-content').style.display = 'block';
        fetchAndDisplayCorrelationMatrix();
    });
});


function fetchFeatureData(displayedFeature) {
    fetch('/feature_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({displayed_feature: displayedFeature}),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
        } else {
            isNumericFeature = data.is_numeric;
            let plotX = data.x;
            let plotY = data.y;

            // Reset the axis layout
            layout.xaxis = {};
            layout.yaxis = {};
            layout.xaxis.title = 'Feature value';
            layout.yaxis.title = {
                text: 'Feature effect on model output',
                standoff: 20
            };

            if (!isNumericFeature) {
                // Set the x-axis layout for categorical data
                layout.xaxis = {
                    tickvals: plotX,
                    ticktext: data.original_values
                };
                Plotly.react('plot', [{
                    x: plotX,
                    y: plotY,
                    type: 'bar'
                }], layout);
            } else {
                // Update the plot for numeric data
                Plotly.react('plot', [{
                    x: plotX,
                    y: plotY,
                    type: 'scatter',
                    mode: 'lines'
                }], layout);
            }
            if (data.hist_data && data.bin_edges) {
                createHistogramPlot(data.hist_data, data.bin_edges);
            }

            store_x_values();
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}



// User options to adjust spline functions

// Updates the plot to set a constant value across a selected range
function setConstantValue() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);
    const newYValue = parseFloat(document.getElementById('new-y-value').value);

    // Store the current x-axis and y-axis range
    const gd = document.getElementById('plot');
    const currentXAxisRange = gd.layout.xaxis.range;
    const currentYAxisRange = gd.layout.yaxis.range;

    fetch('/setConstantValue', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, new_y: newYValue, displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]}).then(() => {
            // Reapply the stored range values to maintain the zoom level
            Plotly.relayout(gd, {
                'xaxis.range': currentXAxisRange,
                'yaxis.range': currentYAxisRange
            });
        });
    });
}

// Updates the plot to set a linear adjustment across a selected range
function setLinear() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    // Store the current x-axis and y-axis range
    const gd = document.getElementById('plot');
    const currentXAxisRange = gd.layout.xaxis.range;
    const currentYAxisRange = gd.layout.yaxis.range;

    // Options: Inplace Interpolation,  Stepwise


    fetch('/setLinear', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]}).then(() => {
            // Reapply the stored range values to maintain the zoom level
            Plotly.relayout(gd, {
                'xaxis.range': currentXAxisRange,
                'yaxis.range': currentYAxisRange
            });
        });
    });
}

// Updates the plot to enforce a monotonic increase across a selected range
function setMonotonicIncrease() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    fetch('/monotonic_increase', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]});
    });
}

// Updates the plot to enforce a monotonic decrease across a selected range
function setMonotonicDecrease() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    fetch('/monotonic_decrease', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]});
    });
}

// Updates the plot to apply smoothing across a selected range
function setSmooth() {
    const x1 = parseFloat(document.getElementById('x1-value').value);
    const x2 = parseFloat(document.getElementById('x2-value').value);

    // Store the current x-axis and y-axis range
    const gd = document.getElementById('plot');
    const currentXAxisRange = gd.layout.xaxis.range;
    const currentYAxisRange = gd.layout.yaxis.range;

    fetch('/setSmooth', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({x1: x1, x2: x2, displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {y: [data.y]}).then(() => {
            // Reapply the stored range values to maintain the zoom level
            Plotly.relayout(gd, {
                'xaxis.range': currentXAxisRange,
                'yaxis.range': currentYAxisRange
            });
        });
    });
}

// Requests spline interpolation for selected features and updates plot
function SplineInterpolation(selectedFeatures) {
    const displayed_feature = document.getElementById('display-feature').value;

    fetch('/cubic_spline_interpolate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ displayed_feature: displayed_feature, selectedFeatures: selectedFeatures })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error occurred: ' + data.error);
        }
        else {
            Plotly.update('plot', {
                y: [data.y]
            }).then(() => {
                predictAndGetMetrics();
            });
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
    });
}



// Sends a request to retrain model with new parameters
function sendRetrainRequest() {

    const displayed_feature = document.getElementById('display-feature').value;
    const elmScaleElement = document.getElementById('hidden_elmScale');
    const elmScale = parseFloat(elmScaleElement.value);
    const elmAlphaElement = document.getElementById('hidden_elmAlpha')
    const elmAlpha = parseFloat(elmAlphaElement.value);
    const nrSyntheticDataPointsElement = document.getElementById('hidden_nrSyntheticDataPoints')
    const nrSyntheticDataPoints = parseInt(nrSyntheticDataPointsElement.value);

    // helpful for debugging
    console.log(JSON.stringify({
        displayed_feature: displayed_feature,
        selectedFeatures: selectedFeatures,
        elmScale: elmScale,
        elmAlpha: elmAlpha,
        nrSyntheticDataPoints: nrSyntheticDataPoints
    }));

    fetch('/retrain_feature', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            displayed_feature: displayed_feature,
            selectedFeatures: selectedFeatures,
            elmScale: elmScale,
            elmAlpha: elmAlpha,
            nrSyntheticDataPoints: nrSyntheticDataPoints
        })
    })
    .then(response => {
        return response.json();
    })
    .then(data => {
        if (data.error) {
            alert('Error occurred: ' + data.error);
        } else {
            Plotly.update('plot', {
                y: [data.y]
            });

        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert(error.message);
    }).then(() => {
        predictAndGetMetrics();
    });
}


// Fetches and displays model prediction metrics
function predictAndGetMetrics() {
    console.log("predicting metrics");
    fetch('/predict_and_get_metrics')
    .then(response => response.json())
    .then(data => {
        let outputContent;
        if (data.task === "regression") {
            outputContent = 'MSE Training: ' + data.train_score.toFixed(2) +
                            '<br>MSE Validation: ' + data.val_score.toFixed(2);
        } else if (data.task === "classification") {
            outputContent = 'F1 Score Training: ' + data.train_score.toFixed(2) +
                            '<br>F1 Score Validation: ' + data.val_score.toFixed(2);
        }
        let OutputDiv = document.getElementById('metric-output');
        OutputDiv.innerHTML = outputContent;
        OutputDiv.style.display = 'block';
    })
    .catch(error => {
        // If there's an error, display it in the output div
        let OutputDiv = document.getElementById('metric-output');
        OutputDiv.className = 'alert alert-danger';
        OutputDiv.innerHTML = 'Error: ' + error;
        OutputDiv.style.display = 'block';
    });
}

// Resets graph to its original state by fetching original data
function resetGraph() {
    fetch('/get_original_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        Plotly.update('plot', {x: [data.x], y: [data.y]});
        predictAndGetMetrics();
    });
}

// Reverts the last change made to the plot (deprecated, see backend)
function undoLastChange() {
    fetch('/undo_last_change', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({displayed_feature: displayedFeature})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            Plotly.update('plot', {y: [data.y]});
        }
    });
}


// Helper functions

// Stores x1 and x2 values of bounding boxes upon selection on plot
function store_x_values() {
    document.getElementById('plot').on('plotly_selected', function(data) {
        if (!data) return;
        const x1 = data.range.x[0];
        const x2 = data.range.x[1];
        document.getElementById('x1-value').value = x1.toFixed(2);
        document.getElementById('x2-value').value = x2.toFixed(2);
    });
};

// Function to generate columns based on JSON keys
function generateColumns(data, backendColumns) {
    var columns = [];
    var backendColumnSet = new Set(backendColumns);
    backendColumns.forEach(columnName => {
        if (columnName === 'ID') return;
        columns.push({
            title: columnName.charAt(0).toUpperCase() + columnName.slice(1),
            field: columnName,
        });
    });

    // Ensure 'ID' column is added first if it exists in backendColumns or in the first row of data
    if (backendColumnSet.has('ID') || data[0]?.hasOwnProperty('ID')) {
        columns.unshift({ title: 'ID', field: 'ID' });
    }
    if (data.length > 0) {
        Object.keys(data[0]).forEach(key => {
            if (!backendColumnSet.has(key)) {
                columns.push({
                    title: key.charAt(0).toUpperCase() + key.slice(1),
                    field: key,
                });
            }
        });
    }
    return columns;
}



let selectedRowId_1 = 0;
let selectedRowId_2 = 0;

// Creates a data grid table with given data and columns
function createTable(data, backendColumns) {
    var table = new Tabulator("#datagrid-table", {
        data: data,
        layout: "fitColumns",
        columns: generateColumns(data, backendColumns),
        // Define the context menu for each row
        rowContextMenu: [
            {
                label:"Order by nearest neighbor",
                action: function (e, row) {
                        e.preventDefault(); // Prevent the browser's context menu from appearing
                        const selectedRowId = row.getData().ID;
                        const tableData = table.getData();
                        console.log(tableData)
                        // Send request to backend
                        fetch('/path/to/order_by_nearest', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                data: tableData,
                                selectedRowId: selectedRowId
                            }),
                        })
                        .then(response => response.json())
                        .then(orderedData => {
                            // Update the table with the newly ordered data
                            table.setData(orderedData);
                        })
                        .catch(error => console.error('Error:', error));
                    }
            },
            {
                label:"Add to plot 1",
                action: function(e, row){
                    e.preventDefault();
                    selectedRowId_1 = row.getData().ID;
                }
            },
            {
                label:"Add to plot 2",
                action: function(e, row){
                    e.preventDefault();
                    selectedRowId_2 = row.getData().ID;
                }
            },
        ],
    });
}

// Fetches data and creates a table for displaying validation data
function fetchDataAndCreateTable() {
    console.log("Method called");
    fetch('/load_data_grid_instances', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(response => {
        console.log(response);
        const { data, columns } = response;
        valDataID = data.map(item => item.ID);
        createTable(data, columns);
    })
    .catch(error => console.error('Error:', error));
}

// Fetches instance explanation and displays it in the UI
function fetchAndDisplayInstanceExplanation() {
    console.log("Fetching Instance Explanation");

    const explanationsContainer = document.getElementById('instance-explanations-content');
    explanationsContainer.innerHTML = '';

    // Create the first dropdown and target display
    const dropdown1 = document.createElement('select');
    dropdown1.innerHTML = valDataID.map(id =>
        `<option value="${id}" ${id === selectedRowId_1 ? 'selected' : ''}>${id}</option>`
    ).join('');
    explanationsContainer.appendChild(dropdown1);

    const targetDisplay1 = document.createElement('span');
    explanationsContainer.appendChild(targetDisplay1);

    const explanation1 = document.createElement('p');
    explanationsContainer.appendChild(explanation1);

    explanationsContainer.appendChild(document.createElement('br'));
    explanationsContainer.appendChild(document.createElement('br'));

    // Create the second dropdown and target display
    const dropdown2 = document.createElement('select');
    dropdown2.innerHTML = valDataID.map(id =>
        `<option value="${id}" ${id === selectedRowId_2 ? 'selected' : ''}>${id}</option>`
    ).join('');
    explanationsContainer.appendChild(dropdown2);

    const targetDisplay2 = document.createElement('span');
    explanationsContainer.appendChild(targetDisplay2);

    const explanation2 = document.createElement('p');
    explanationsContainer.appendChild(explanation2);

    // Event listeners for dropdowns
    dropdown1.addEventListener('change', function() {
        selectedRowId_1 = this.value;
        fetchExplanation(selectedRowId_1, explanation1, targetDisplay1);
    });

    dropdown2.addEventListener('change', function() {
        selectedRowId_2 = this.value;
        fetchExplanation(selectedRowId_2, explanation2, targetDisplay2);
    });

    // Initial fetch for the first load
    fetchExplanation(selectedRowId_1, explanation1, targetDisplay1);
    fetchExplanation(selectedRowId_2, explanation2, targetDisplay2);
}

// Fetches instance explanation for a selected row ID
function fetchExplanation(selectedRowId, chartContainer, targetDisplay) {
    fetch('/instance_explanation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({selectedRow_ID: selectedRowId})
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(explanationData => {
        targetDisplay.innerHTML = `     ID: ${selectedRowId}, Prediction: ${explanationData.prediction}, 
        Target: ${explanationData.target}, Intercept: <span style="color: red;">${explanationData.intercept}</span>`;

        if (chartContainer.firstChild) {
            chartContainer.removeChild(chartContainer.firstChild);
        }

        chartContainer.style.maxWidth = '450px';
        chartContainer.style.height = 'auto'; //


        const canvas = document.createElement('canvas');
        chartContainer.appendChild(canvas);

        const ctx = canvas.getContext('2d');
        const myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: explanationData.feature_names,
                datasets: [{
                    label: 'Prediction Contribution',
                    data: explanationData.pred_instance,
                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: explanationData.intercept,
                                yMax: explanationData.intercept,
                                borderColor: 'red',
                                borderWidth: 2,
                                label: {
                                    enabled: false,
                                    content: 'Intercept',
                                    position: 'end'
                                }
                            }
                        }
                    }
                }
            }
        });
    })
    .catch(error => console.error('Error:', error));
}

// Fetches and displays shape functions as an image

function fetchAndDisplayShapeFunctions() {
    const shapeFunctionsContainer = document.getElementById('shape-functions-content');
    shapeFunctionsContainer.innerHTML = '<p>Loading shape functions...</p>';
    shapeFunctionsContainer.style.textAlign = 'center';

    fetch('/plot_all_shape_functions')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(imageBlob => {
            const imageObjectURL = URL.createObjectURL(imageBlob);
            // Display the image
            shapeFunctionsContainer.innerHTML = `<img src="${imageObjectURL}" alt="Shape Functions" style="max-width: 100%; height: auto;">`;
        })
        .catch(error => {
            console.error('Error fetching the shape functions image:', error);
            shapeFunctionsContainer.innerHTML = '<p>Error loading shape functions.</p>';
        });
}



// Function to fetch and display correlation matrix
function fetchAndDisplayCorrelationMatrix() {
    const correlationMatrixContainer = document.getElementById('correlation-matrix-content');
    correlationMatrixContainer.innerHTML = '<p>Loading correlation matrix...</p>';
    correlationMatrixContainer.style.textAlign = 'center';


    fetch('/plot_correlation_matrix')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(imageBlob => {
            const imageObjectURL = URL.createObjectURL(imageBlob);

            // Display the image
            correlationMatrixContainer.innerHTML = `<img src="${imageObjectURL}" alt="Correlation Matrix" style="max-width: 100%; height: auto;">`;
        })
        .catch(error => {
            console.error('Error fetching the correlation matrix image:', error);
            correlationMatrixContainer.innerHTML = '<p>Error loading correlation matrix.</p>';
        });
}

// Hides all content sections
function hideAllContentSections() {
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
}



