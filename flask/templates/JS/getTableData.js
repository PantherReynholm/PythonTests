function showtableData (tableID, calorieColumnID) {

    document.getElementById(buttonID).innerHTML = "Calculate Calories Burned"
    var myTab = document.getElementById(tableID)

    for (i=1; i < myTab.rows.length; i++) {
        
        var objCells = myTab.rows.item(i).cells;

        for (var j = 0; j < objCells.length; j++) {
            tableID.innerHTML = tableID.innerHTML + " " + objCells.item(i).innerHTML;
        }
        calorieColumnID.innerHTML = tableID.innerHTML + "<br />";
    }
}