from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Cargar conjunto de datos
titulares = [
    "mataron al presidente de EEUU",
    "nuevo tratamiento para el cancer",
    "EE.UU. aprueba paquete de estímulo económico de 1,9 billones de dólares",
    "Pfizer anuncia que su vacuna contra el COVID-19 tiene una efectividad del 95%",
    "El presidente de Brasil, Jair Bolsonaro, se recupera del COVID-19",
    "Microsoft adquiere GitHub por 7.500 millones de dólares",
    "La NASA descubre agua en la luna",
    "Se descubre posible cura para el Alzheimer",
    "El precio del petróleo cae a su nivel más bajo en años",
    "Primer vuelo tripulado a Marte previsto para 2026",
    "Inauguran la torre más alta del mundo en Dubai",
    "Elon Musk anuncia que su empresa enviará turistas a la luna en 2023",
    "Acuerdo histórico entre Israel y Emiratos Árabes Unidos",
    "Terremoto de magnitud 7,5 sacude la costa de México",
    "Nueva cepa del COVID-19 detectada en Sudáfrica",
    "China lanza satélite para investigación del cambio climático",
    "El volcán más grande de Islandia entra en erupción",
    "Fallece famoso actor de Hollywood a los 89 años",
    "Confirman existencia de agua líquida en Marte",
    "Compañía aérea anuncia la eliminación de todos sus vuelos de larga distancia",
    "Descubren nueva especie de dinosaurio en Argentina",
    "Campeonato mundial de fútbol se realizará en Qatar en 2022",
    "Primera misión tripulada a la Estación Espacial China",
    "Fuertes lluvias causan inundaciones en varias ciudades del mundo",
    "Crisis en Venezuela: miles de personas huyen del país en busca de un futuro mejor",
    "Estados Unidos impone nuevas sanciones a Venezuela",
    "Científicos descubren nueva especie de dinosaurio en Argentina",
    "Apple lanza su nuevo modelo de iPhone",
    "Mujer se gradúa a los 93 años de edad",
    "La bolsa de valores registra una caída histórica",
    "El Real Madrid gana la liga de fútbol",
    "Nuevos hallazgos arqueológicos en Egipto",
    "Investigadores encuentran posible cura para el Alzheimer",
    "El huracán Laura azota la costa del Golfo de México",
    "La NASA descubre agua en la luna",
    "China lanza su primera misión a Marte",
    "La economía mexicana se recupera más rápido de lo esperado",
    "Descubren restos de naufragio de hace más de 400 años",
    "La OMS declara que el brote de ébola en Congo es de alto riesgo",
    "La empresa de Elon Musk lanza su primer cohete tripulado al espacio",
    "Nuevo récord de contagios de COVID-19 en India",
    "Argentina legaliza el aborto",
    "La ciudad de Nueva York anuncia plan para reabrir escuelas",
    "Descubren nueva especie de planta en la selva amazónica",
    "Elon Musk se convierte en la persona más rica del mundo"
    ]
comentarios = [
    "me gusta la pizza",
    "la pelicula fue aburrida",
    "Me gusta mucho el chocolate",
    "La película que vi anoche fue muy aburrida",
    "Mi deporte favorito es el fútbol",
    "Hoy hizo mucho frío en mi ciudad",
    "Estoy muy contento porque me gradué de la universidad",
    "La película que vi anoche fue muy aburrida",
    "El clima hoy está muy agradable",
    "Estoy cansado después de un largo día de trabajo",
    "Me encanta la música de los años 80",
    "La cena en el restaurante estuvo deliciosa",
    "Hoy me desperté temprano y pude disfrutar del amanecer",
    "Me siento muy feliz de haber encontrado trabajo",
    "Acabo de leer un libro muy interesante",
    "Me encanta viajar y conocer nuevas culturas",
    "Estoy aburrido, no sé qué hacer",
    "No me gusta el ruido de la ciudad, prefiero la tranquilidad del campo",
    "Me gusta mucho el chocolate",
    "Estoy preocupado por el cambio climático",
    "Hace mucho tiempo que no veo a mi familia",
    "No puedo esperar para ir de vacaciones a la playa",
    "Hoy hice ejercicio en el gimnasio y me siento muy bien",
    "Estoy triste porque mi equipo favorito perdió el partido",
    "La obra de teatro a la que fui ayer fue excelente",
    "Estoy muy emocionado porque me voy a casar el próximo mes",
    "Me encanta el clima de esta ciudad",
    "El servicio en este restaurante es excelente",
    "Esta canción me hace sentir nostálgico",
    "No soporto el tráfico en esta ciudad",
    "El último libro que leí fue muy emocionante",
    "El partido de ayer fue aburrido",
    "La atención al cliente de esta compañía es pésima",
    "Odio tener que hacer fila en el supermercado",
    "El nuevo juego de video es increíble",
    "Este programa de televisión es muy entretenido",
    "No me gusta el sabor de esta bebida",
    "Este lugar tiene una vista espectacular",
    "El vestido que compré ayer es hermoso",
    "No puedo creer que ganó esa persona en las elecciones",
    "El concierto de anoche estuvo increíble",
    "Esta película es muy conmovedora",
    "No entiendo por qué tanta gente habla de esa serie de televisión",
    "Este producto es de mala calidad",
    "Me encanta este parque, es muy relajante",
    "El trabajo que tengo es muy estresante"
    ]
X = titulares + comentarios
y = ['titular']*len(titulares) + ['comentario']*len(comentarios)

# Preprocesamiento de datos
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Dividir conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo SVM
clf = SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train, y_train)