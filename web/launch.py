from main import Neuron
import json
import cherrypy
import datetime

neuronApple = Neuron("AAPL.csv")
neuronAmazon = Neuron("AMZN.csv")
neuronGoogle = Neuron("GOOG.csv")


class NeuronApi(object):

    @cherrypy.expose
    def index(self):
        return """<html>
          <body>
            <form method="get" action="stock?company=Amazon">
              <button value="Amazon" name="company" type="submit">Amazon</button>
            </form>
            <form method="get" action="stock?company=Apple">
              <button value="Apple" name="company" type="submit">Apple</button>
            </form>
            <form method="get" action="stock">
              <button value="Google" name="company" type="submit">Google</button>
            </form>
            
            
            <form method="get" action="stock">
              <input type="text" value="Amazon" name="company" />
              <button type="submit">Search</button>
            </form>
            
            
          </body>
        </html>"""

    @cherrypy.expose
    def stock(self, company):

        match company:
            case "Amazon":
                neuron = neuronAmazon
            case "Apple":
                neuron = neuronApple
            case "Google":
                neuron = neuronGoogle
            case _:
                return "К сожалению в нашем списке нет такой компании!"

        tabl = "[['Дата', '" + company + "' ]," + neuron.dots + "]"
        print(neuron.will_it_grow)

        if neuron.will_it_grow:
            will = """<font color="green" size="8">Сегодня можно купить акции """ + company + """, так как они скорее всего завтра вырастут! 
        Предпологаемая цена """ + str(round(neuron.result, 2)) + """ $</font> """
        else:
            will = """<font color="red" size="8">Сегодня покупать акции """ + company + """ не стоит, так как они скорее всего завтра упадут! 
        Предпологаемая цена """ + str(round(neuron.result, 2)) + """ $</font> """

        return """
<html>
 <head>
  <meta charset="utf-8">
  <title>""" + company + """-Акции</title>
  <script src="https://www.google.com/jsapi"></script>
  <script>
   google.load("visualization", "1", {packages:["corechart"]});
   google.setOnLoadCallback(drawChart);
   function drawChart() {
    var data = google.visualization.arrayToDataTable(""" + tabl + """);
    var options = {
     title: 'Стоимость акций компании """ + company + """ за последние 30 дней. (""" + str(
            datetime.datetime.now().date()) + """)',
     hAxis: {title: 'Дата'},
     vAxis: {title: 'Стоимость в $'}
    };
    var chart = new google.visualization.ColumnChart(document.getElementById('oil'));
    chart.draw(data, options);
   }
  </script>
 </head>
 <body>
    <div id="outer">
        <center>
            <div id="oil" style="width: 1600px; height: 600px;"></div>    
        </center>
    </div>
  
  <div>
        """ + will + """
  </div>
  
  <div>
        <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">Купить акции</a>
  </div>
 </body>
 </html>
        """

    # @cherrypy.expose
    # def getjson(self):
    #     now = datetime.datetime.now()
    #     data = {
    #         'companyName': 'Apple',
    #         'date': str(now.day),
    #         'willItGrow': neuronApple.will_it_grow,
    #         'result': str(neuronApple.result)
    #     }
    #     return json.dumps(data)


if __name__ == '__main__':
    cherrypy.quickstart(NeuronApi())
