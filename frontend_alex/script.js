    google.charts.load('current', {'packages':['corechart']});
    google.charts.setOnLoadCallback(drawChart);

    function drawChart() {
		var data = google.visualization.arrayToDataTable([
			['', 'Значение X', 'Значение Y', 'Значение Z'		 ],
			['',  166.753021,  165.945343,	165.4722872956936],
			['',  165.945343,  161.328537,	165.33569094733014],
			['',  161.328537,  162.415436, 	162.8932983771043],
			['',  162.415436,  156.352783,	162.00118790741868]
        ]);

        var options = {
			title: 'Cryptocurrency exchange rate',
			legend: { position: 'absolute' }
        };

        var chart = new google.visualization.LineChart(document.getElementById('curve_chart'));

        chart.draw(data, options);
    }
	  
	
	  