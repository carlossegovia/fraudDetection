import json
import re
from datetime import datetime


class TransformDataset():

    @staticmethod
    def transform(old_file, new_file, map_file):
        # Los features originales
        features = ["monto", "importe_impuesto", "moneda", "unidades_tasadas", "codigo_concep", "fecha", "origen",
                    "destino",
                    "fecha_inicio", "fecha_fin", "duracion", "troncal_entrada", "troncal_salida", "cobro_revertido",
                    "tipo", "tipo_destino", "cod_pais_origen", "cod_pais_destino",
                    "cod_area_destino", "numero_destino", "pais_destino", "tipo_horario"]
        # Features finales
        final_features = ["monto", "importe_impuesto", "unidades_tasadas", "codigo_concep", "origen", "destino",
                          "duracion", "troncal_entrada", "troncal_salida", "cobro_revertido", "tipo", "tipo_destino",
                          "cod_pais_origen", "cod_pais_destino", "cod_area_destino", "tipo_horario",
                          "dia_semana", "hora_llamada"]

        # Features que se descartan
        not_copy = [features.index("fecha"), features.index("moneda"), features.index("fecha_inicio"),
                    features.index("fecha_fin"), features.index("numero_destino"),
                    features.index("tipo_horario"), features.index("pais_destino")]
        # Se mapean estos features porque son de tipo string
        string_features = {
            "codigo_concep": [], "troncal_entrada": [], "troncal_salida": [], "cobro_revertido": [], "tipo_destino": [],
            "tipo_horario": []
        }

        i = 0
        f = 0
        c = 0
        with open(old_file) as f1, open(new_file, "w+") as f2:
            for line in f1:
                datos = line.split(",")
                if len(datos) == 22:
                    new_data = [x for x in datos if datos.index(x) not in not_copy]
                    # se eliminan caracteres basura
                    new_data.append(datos[features.index("tipo_horario")].rstrip().replace("|", ""))
                    # fecha
                    fecha = datetime.strptime(datos[features.index("fecha")], "%Y-%m-%d %H:%M:%S.%f")
                    # fecha de la semana
                    new_data.append(str(fecha.weekday()))
                    # hora de la llamada
                    new_data.append(str(fecha.hour))
                    # mapear a un indice
                    for key in string_features:
                        value = new_data[final_features.index(key)]
                        if value not in string_features[key]:
                            string_features[key].append(value)
                        new_data[final_features.index(key)] = str(string_features[key].index(value))
                    # se eliminan los caracteres no numericos
                    # new_line = re.sub('[<>;a-zA-Z]+', '', ",".join(new_data)) [0-9.,]+
                    new_line = re.sub('[^,.0-9]+', '', ",".join(new_data))
                    if "" not in new_line.split(","):
                        # se escribe en el nuevo archivo
                        f += 1
                        f2.write(new_line + "\n")
                    else:
                        c += 1
                else:
                    i += 1

        # guardar el diccionario en un archivo
        open(map_file, "w+").write(json.dumps(string_features, ensure_ascii=False))
        print "Cantidad de filas que no cumplen con los features necesarios (22): {}".format(i)
        print "Cantidad de filas con espacio vacio: {}".format(c)
        print "Cantidad de filas totales en el dataset: {}".format(f)


if __name__ == "__main__":
    # Nombre del archivo a transformar
    old_file_name = "llamadas_062018.csv"
    # Nombre del archivo de salida
    new_file_name = "transformed_data_mes06.csv"
    # Nombre del archivo auxiliar
    map_file_name = "map_data_mes06.json"
    # Directorio en donde se almacenan los archivos
    directory = "/home/carlitos/Descargas/Tesis/"
    old_file = "{}{}".format(directory, old_file_name)
    new_file = "{}{}".format(directory, new_file_name)
    map_file = "{}{}".format(directory, map_file_name)
    TransformDataset.transform(old_file, new_file, map_file)
