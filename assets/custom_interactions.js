
document.addEventListener('dblclick', function(e) {
    if (e.target.classList.contains('dblclick-enlarge')) {
        // Find the store component
        // Depending on how dash renders, we might need to verify where to set props.
        // Dash 2.11+ supports set_props via window.dash_clientside.set_props(id, props)
        
        // We will assume window.dash_clientside.set_props is available.
        // Requires Dash >= 2.9 (strictly >= 2.11 for no_update safety, but okay)
        
        var src = e.target.src;
        if (src && window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props('double-click-store', {data: src});
        } else {
            console.warn("Dash clientside set_props not available or src missing.");
        }
    }
});
