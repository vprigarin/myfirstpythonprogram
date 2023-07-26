# Моя первая программа на Питоне

решает уравнение
        A<sub>x,y,t<sub>d</sub>,t<sub>s</sub></sub> X<sub>x,y,t<sub>s</sub></sub> = n<sub>t<sub>d</sub></sub>
методом наименьших квадратов.
Решение X ограничено условием неотрицательности X<sub>*</sub> >= 0, поскольку речь идет о числе частиц.
В стандартной библиотеке SciPy наличествуют две реализации этого метода: <strong>nnls</strong> и <strong>lsq_linear</strong>. В последнем реализованы два метода решения и возможно задание разнообразных ограничений.
Как выяснилось в процессе, <strong>nnls</strong> по каким-то причинам не справляется с данной задачей. Также, из двух методов в <strong>lsq_linear</strong>, работает только один. Это довольно странно, учитывая, что реализации заимствовались, скорее всего, из LAPACK или MINPACK, где всё работает как часы, так сказать, издревле. Тем более, что в SciKit-Learn и, как я понял, в TensorFlow тоже, реализации метода наименьших квадратов основаны на <strong>nnls</strong>.

## Вернемся к заданию.

<strong>lsq_linear</strong> сработал, невязки во всех случаях (см.ниже) оказались порядка 10<sup>-12</sup>, что неудивительно для недоопределенной задачи.
Как можно оценить координату источника "через точку минимальной невязки", я, честно говоря не понял. Невязки чего? Уравнения? Так они на уровне ошибок округления, это не победить. Время начала испускания частиц также не было представлено в условиях.
Поскольку речь в задаче идет о переносе частиц, логично предположить, что величины в последовательные моменты времени коррелированы. Попытка оценить <em>средний</em> коэффициент корреляции (процедура Хилдрета—Лу, Дарбина - вообще-то, авторов там гораздо больше; всё это открывалось и переоткрывалось неоднократно) к успеху не привела. Результаты практически не меняются для разных r, что, опять же, можно списать на недоопределенность задачи. По той же причине (нулевые остатки), не проводилась и коррекция через регрессию остатков и обобщенный МНК.
В качестве ответа выдается серия картинок восстановленого значения X<sub>t<sub>s</sub></sub> для t<sub>s</sub> = 0..9 и разных значениях коэффициента автокорреляции.
Возможно, из картинок удастся получить какую-то полезную информацию.

## О тестировании

Не понял. Стандартную библиотеку? Ну вот, два из трех опробованных вариантов не сработали.
Стандартная же процедура: реализуем, по возможности, всё. Оставляем выживших.
Полновесное тестирование метода решения выходит за рамки тестового задания.
Это, впрочем, можно обсудить.

## CUDA и распараллеливание

Где-то я читал, что существует вариант NumPy, скомпилированный с Intel MKL (скорее всего, не бесплатный). В MKL встроены TBB (многоядерность) и/или OpenCL (многоядерность и графические ускорители). Не нашел сходу.
Была надежда, что в TensorFlow найдется реализация сингулярного разложения хотя бы для CUDA, но увы.
Еще я надеялся потестировать разложение на основе степенного алгоритма - почто 100% параллелизуется. Мне почему-то,- по статьям, конечно,- казалось, что реализация дошла до питоновских библиотек. Нет. В наличии только для MATLAB'a.
Может, я плохо искал, но в общем, впечатление довольно удручающее.

Реализовывать и отрабатывать эти алгоритмы - задача явно не для тестового задания.
Втыкать известные реализации библиотек линейной алгебры (типа Paralution)- ну зачем это для теста. Они все на С++,
нехота заморачиваться с интерфейсными проблемами, ведь оберткой только для одной функции тут не отделаешься.

# Запуск

Не будучи специалистом по Питону, могу только пересказать инструкцию:
 - установите Anaconda по инструкции, дальнейшие действия в командной строке.
 - установите тестовую среду: conda env create -f schooltech.yml.Сам файл в корневом каталоге.
 - запустите jupyter lab и загрузите Learn/arrays.ipynb
 - альтернативно, запустите spyder и загрузите assignment.py
 - альтернативно, выполните команду python assignment.py. В этом случае картинки придется убивать вручную, что неудобно. Извините, я не успел разобраться.

Содержимое Learn/arrays.ipynb и assignment.py совпадает.

# Заключительные соображения

Благодарю за повод поизучать Питон. Может и пригодится.
