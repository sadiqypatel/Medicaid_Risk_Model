# Databricks notebook source
from pyspark.sql.functions import col,isnan, when, count, desc, concat, expr, array, struct, expr, lit, col, concat, substring, array, explode, exp, expr, sum, round, mean, posexplode, first, udf
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import count
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
import numpy as np

# COMMAND ----------

df = spark.table("dua_058828_spa240.stage2_cost_analysis_sample")
# Show the merged DataFrame
print((df.count(), len(df.columns)))
print(df.printSchema())

# COMMAND ----------

columns = ['beneID', 'state', 'ageCat', 'sex', 'ccsr_null', 'BLD001', 'BLD002', 'BLD003', 'BLD004', 'BLD005', 'BLD006', 'BLD007', 'BLD008', 'BLD009', 'BLD010', 'CIR001', 'CIR002', 'CIR003', 'CIR004', 'CIR005', 'CIR006', 'CIR007', 'CIR008', 'CIR009', 'CIR010', 'CIR011', 'CIR012', 'CIR013', 'CIR014', 'CIR015', 'CIR016', 'CIR017', 'CIR018', 'CIR019', 'CIR020', 'CIR021', 'CIR022', 'CIR023', 'CIR024', 'CIR025', 'CIR026', 'CIR027', 'CIR028', 'CIR029', 'CIR030', 'CIR031', 'CIR032', 'CIR033', 'CIR034', 'CIR035', 'CIR036', 'CIR037', 'CIR038', 'CIR039', 'DEN001', 'DIG001', 'DIG004', 'DIG005', 'DIG006', 'DIG007', 'DIG008', 'DIG009', 'DIG010', 'DIG011', 'DIG012', 'DIG013', 'DIG014', 'DIG015', 'DIG016', 'DIG017', 'DIG018', 'DIG019', 'DIG020', 'DIG021', 'DIG022', 'DIG023', 'DIG024', 'DIG025', 'EAR001', 'EAR002', 'EAR003', 'EAR004', 'EAR005', 'EAR006', 'END001', 'END002', 'END003', 'END007', 'END008', 'END009', 'END010', 'END011', 'END012', 'END013', 'END014', 'END015', 'END016', 'END017', 'EXT001', 'EXT002', 'EXT003', 'EXT004', 'EXT005', 'EXT006', 'EXT007', 'EXT008', 'EXT009', 'EXT010', 'EXT011', 'EXT012', 'EXT013', 'EXT014', 'EXT015', 'EXT016', 'EXT017', 'EXT018', 'EXT019', 'EXT025', 'EXT026', 'EXT027', 'EXT028', 'EXT029', 'EXT030', 'EYE001', 'EYE002', 'EYE003', 'EYE004', 'EYE005', 'EYE006', 'EYE007', 'EYE008', 'EYE009', 'EYE010', 'EYE011', 'EYE012', 'FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005', 'FAC006', 'FAC007', 'FAC008', 'FAC009', 'FAC010', 'FAC011', 'FAC012', 'FAC013', 'FAC014', 'FAC015', 'FAC016', 'FAC017', 'FAC018', 'FAC019', 'FAC020', 'FAC021', 'FAC022', 'FAC023', 'FAC024', 'FAC025', 'GEN001', 'GEN002', 'GEN003', 'GEN004', 'GEN005', 'GEN006', 'GEN007', 'GEN008', 'GEN009', 'GEN010', 'GEN011', 'GEN012', 'GEN013', 'GEN014', 'GEN015', 'GEN016', 'GEN017', 'GEN018', 'GEN019', 'GEN020', 'GEN021', 'GEN022', 'GEN023', 'GEN024', 'GEN025', 'GEN026', 'INF001', 'INF002', 'INF003', 'INF004', 'INF005', 'INF006', 'INF007', 'INF008', 'INF009', 'INF010', 'INF011', 'INJ001', 'INJ002', 'INJ003', 'INJ004', 'INJ005', 'INJ006', 'INJ007', 'INJ008', 'INJ009', 'INJ010', 'INJ011', 'INJ012', 'INJ013', 'INJ014', 'INJ015', 'INJ016', 'INJ017', 'INJ018', 'INJ019', 'INJ021', 'INJ024', 'INJ025', 'INJ026', 'INJ027', 'INJ028', 'INJ029', 'INJ030', 'INJ031', 'INJ032', 'INJ033', 'INJ034', 'INJ035', 'INJ036', 'INJ037', 'INJ038', 'INJ039', 'INJ040', 'INJ041', 'INJ042', 'INJ043', 'INJ044', 'INJ045', 'INJ046', 'INJ047', 'INJ048', 'INJ049', 'INJ050', 'INJ051', 'INJ052', 'INJ053', 'INJ054', 'INJ055', 'INJ056', 'INJ057', 'INJ058', 'INJ059', 'INJ060', 'INJ061', 'INJ062', 'INJ063', 'INJ064', 'INJ065', 'INJ066', 'INJ067', 'INJ068', 'INJ069', 'INJ070', 'INJ071', 'INJ072', 'INJ073', 'INJ074', 'INJ075', 'INJ076', 'MAL001', 'MAL002', 'MAL003', 'MAL004', 'MAL005', 'MAL006', 'MAL007', 'MAL008', 'MAL009', 'MAL010', 'MBD001', 'MBD002', 'MBD003', 'MBD004', 'MBD005', 'MBD006', 'MBD007', 'MBD008', 'MBD009', 'MBD010', 'MBD011', 'MBD012', 'MBD013', 'MBD014', 'MBD017', 'MBD018', 'MBD019', 'MBD020', 'MBD021', 'MBD022', 'MBD023', 'MBD024', 'MBD025', 'MBD026', 'MUS001', 'MUS002', 'MUS003', 'MUS004', 'MUS005', 'MUS006', 'MUS007', 'MUS008', 'MUS009', 'MUS010', 'MUS011', 'MUS012', 'MUS013', 'MUS014', 'MUS015', 'MUS016', 'MUS017', 'MUS018', 'MUS019', 'MUS020', 'MUS021', 'MUS022', 'MUS023', 'MUS024', 'MUS025', 'MUS026', 'MUS028', 'MUS030', 'MUS031', 'MUS032', 'MUS033', 'MUS034', 'MUS036', 'MUS037', 'MUS038', 'NEO001', 'NEO002', 'NEO003', 'NEO004', 'NEO005', 'NEO006', 'NEO007', 'NEO008', 'NEO009', 'NEO010', 'NEO011', 'NEO012', 'NEO013', 'NEO014', 'NEO015', 'NEO016', 'NEO017', 'NEO018', 'NEO019', 'NEO020', 'NEO021', 'NEO022', 'NEO023', 'NEO024', 'NEO025', 'NEO026', 'NEO027', 'NEO028', 'NEO029', 'NEO030', 'NEO031', 'NEO032', 'NEO033', 'NEO034', 'NEO035', 'NEO036', 'NEO037', 'NEO038', 'NEO039', 'NEO040', 'NEO041', 'NEO042', 'NEO043', 'NEO044', 'NEO045', 'NEO046', 'NEO047', 'NEO048', 'NEO049', 'NEO050', 'NEO051', 'NEO052', 'NEO053', 'NEO054', 'NEO055', 'NEO056', 'NEO057', 'NEO058', 'NEO059', 'NEO060', 'NEO061', 'NEO062', 'NEO063', 'NEO064', 'NEO065', 'NEO066', 'NEO067', 'NEO068', 'NEO069', 'NEO070', 'NEO071', 'NEO072', 'NEO073', 'NEO074', 'NVS001', 'NVS002', 'NVS003', 'NVS004', 'NVS005', 'NVS006', 'NVS007', 'NVS008', 'NVS009', 'NVS010', 'NVS011', 'NVS012', 'NVS013', 'NVS014', 'NVS015', 'NVS016', 'NVS017', 'NVS018', 'NVS019', 'NVS020', 'NVS021', 'NVS022', 'PNL001', 'PNL002', 'PNL003', 'PNL004', 'PNL005', 'PNL006', 'PNL007', 'PNL008', 'PNL009', 'PNL010', 'PNL011', 'PNL012', 'PNL013', 'PNL014', 'PRG001', 'PRG002', 'PRG003', 'PRG004', 'PRG005', 'PRG006', 'PRG007', 'PRG008', 'PRG009', 'PRG010', 'PRG011', 'PRG012', 'PRG013', 'PRG014', 'PRG015', 'PRG016', 'PRG017', 'PRG018', 'PRG020', 'PRG021', 'PRG022', 'PRG023', 'PRG024', 'PRG025', 'PRG026', 'PRG027', 'PRG028', 'PRG029', 'PRG030', 'RSP001', 'RSP002', 'RSP003', 'RSP004', 'RSP005', 'RSP006', 'RSP007', 'RSP008', 'RSP009', 'RSP010', 'RSP011', 'RSP012', 'RSP013', 'RSP014', 'RSP015', 'RSP016', 'RSP017', 'SKN001', 'SKN002', 'SKN003', 'SKN004', 'SKN005', 'SKN006', 'SKN007', 'SYM001', 'SYM002', 'SYM003', 'SYM004', 'SYM005', 'SYM006', 'SYM007', 'SYM008', 'SYM009', 'SYM010', 'SYM011', 'SYM012', 'SYM013', 'SYM014', 'SYM015', 'SYM016', 'SYM017', 'rx_null', 'E01754130101', 'E01754140101', 'E01754150101', 'E01754160101', 'E01754180101', 'E01754180201', 'E01754190101', 'E01754190201', 'E01754200101', 'E01754200201', 'E01754210101', 'E01754210201', 'E01754230101', 'E01754260101', 'E01754260401', 'E01754270101', 'E01754280101', 'E01754290101', 'E01754300101', 'E01754300201', 'E01754310101', 'E01754340101', 'E01754350101', 'E01754350201', 'E01754430101', 'E01754430201', 'E01754530101', 'E01754540101', 'E01754570101', 'E01754610101', 'E01754620101', 'E01754630101', 'E01754630201', 'E01754630202', 'E01754660101', 'E01754660301', 'E01754670101', 'E01754680101', 'E01754700101', 'E01754730101', 'E01754760101', 'E01754770101', 'E01754770201', 'E01754770202', 'E01754770301', 'E01754810101', 'E01754820101', 'E01754820201', 'E01754830101', 'E01754830201', 'E01754840101', 'E01754850101', 'E01754860101', 'E01754870101', 'E01754870201', 'E01754880101', 'E01754880201', 'E01754890101', 'E01754890201', 'E01754890202', 'E01754910101', 'E01754930101', 'E01754940101', 'E01754950101', 'E01754960101', 'E01754960201', 'E01754970101', 'E01754970201', 'E01754980101', 'E01754990101', 'E01755000101', 'E01755010101', 'E01755030101', 'E01755050101', 'E01755070101', 'E01755090101', 'E01755110101', 'E01755140101', 'E01755150101', 'E01755170101', 'E01755170201', 'E01755190101', 'E01755210101', 'E01755220101', 'E01755240101', 'E01755250101', 'E01755350101', 'E01755390101', 'E01755430101', 'E01755520101', 'E01755530101', 'E01755540101', 'E01755550101', 'E01755560101', 'E01755570101', 'E01755570201', 'E01755580101', 'E01755590101', 'E01755600101', 'E01755610101', 'E01755610201', 'E01755610202', 'E01755610203', 'E01755610301', 'E01755620101', 'E01755620201', 'E01755620202', 'E01755620203', 'E01755630101', 'E01755630201', 'E01755640101', 'E01755640201', 'E01755650101', 'E01755650201', 'E01755650202', 'E01755650203', 'E01755650401', 'E01755660101', 'E01755680101', 'E01755700101', 'E01755720101', 'E01755730101', 'E01755740101', 'E01755740201', 'E01755740202', 'E01755740203', 'E01755740301', 'E01755740303', 'E01755750101', 'E01755760101', 'E01755760201', 'E01755760202', 'E01755760203', 'E01755760204', 'E01755760206', 'E01755760207', 'E01755760209', 'E01755760302', 'E01755760401', 'E01755780101', 'E01755790101', 'E01755790201', 'E01755800101', 'E01755810101', 'E01755820101', 'E01755830101', 'E01755840101', 'E01755860101', 'E01755870101', 'E01755870201', 'E01755870202', 'E01755880101', 'E01755890101', 'E01755900101', 'E01755910101', 'E01755920101', 'E01755940101', 'E01755940401', 'E01755940501', 'E01755950101', 'E01755960101', 'E01755960401', 'E01755970101', 'E01755980101', 'E01755990101', 'E01756010101', 'E01756020101', 'E01756030101', 'E01756040101', 'E01756050101', 'E01756060101', 'E01756070101', 'E01756080101', 'E01756090101', 'E01756100101', 'E01756110101', 'E01756120101', 'E01756130101', 'E01756160101', 'E01756230101', 'E01756250101', 'E01756300101', 'E01756340101', 'E01756370101', 'E01756380101', 'E01756550101', 'E01756550201', 'E01756560101', 'E01756570101', 'E01756570201', 'E01756610101', 'E01756640101', 'E01756650101', 'E01756660101', 'E01756670101', 'E01756690101', 'E01756790101', 'E01756800101', 'E01756810101', 'E01756820101', 'E01756820201', 'E01756820301', 'E01756820302', 'E01756820501', 'E01756830101', 'E01756890101', 'E01756900101', 'E01756900201', 'E01756910101', 'E01756910201', 'E01756910202', 'E01756920101', 'E01756930101', 'E01756930501', 'E01756940101', 'E01756940201', 'E01756950101', 'E01756960101', 'E01756980101', 'E01757000101', 'E01757000201', 'E01757050101', 'E01757060101', 'E01757100101', 'E01757120101', 'E01757130101', 'E01757160101', 'E01757190101', 'E01757200101', 'E01757220101', 'E01757220201', 'E01757220202', 'E01757220203', 'E01757220204', 'E01757220205', 'E01757220206', 'E01757220207', 'E01757230101', 'E01757230201', 'E01757370101', 'E01757370301', 'E01757380101', 'E01757390101', 'E01757390301', 'E01757400101', 'E01757430101', 'E01757440101', 'E01757450101', 'E01757460101', 'E01757460201', 'E01757460202', 'E01757460301', 'E01757490101', 'E01757500101', 'E01757500201', 'E01757510101', 'E01757520101', 'E01757530101', 'E01757530201', 'E01757540101', 'E01757570101', 'E01757580101', 'E01757590101', 'E01757650101', 'E01757660101', 'E01757680101', 'E01757690101', 'E01757710101', 'E01757750101', 'E01757760101', 'E01757770101', 'E01757790101', 'E01757800101', 'E01757810101', 'E01757820101', 'E01757830101', 'E01757840101', 'E01757840201', 'E01757850101', 'E01757860101', 'E01757900101', 'E01757940101', 'E01757960101', 'E01758000101', 'E01758010101', 'E01758020101', 'E01758050101', 'E01758070101', 'E01758090101', 'E01758100101', 'E01758110101', 'E01758110201', 'E01758120101', 'E01758140101', 'E01758150101', 'E01758180101', 'E01758180201', 'E01758190101', 'E01758200101', 'E01758210101', 'E01758220101', 'E01758230101', 'E01758240101', 'E01758250101', 'E01758250201', 'E01758250203', 'E01758250302', 'E01758260101', 'E01758270101', 'E01758280101', 'E01758310101', 'E01758350101', 'E01758360101', 'E01758360201', 'E01758370101', 'E01758380101', 'E01758390101', 'E01758400101', 'E01758410101', 'E01758420101', 'E01758430101', 'E01758450101', 'E01758470101', 'E01758480101', 'E01758490101', 'E01758540101', 'E01758740101', 'E01758750101', 'E01758770101', 'E01758770201', 'E01758790101', 'E01758810101', 'E01758820101', 'E01758840101', 'E01758850101', 'E01758870101', 'E01758870201', 'E01758890101', 'E01758950101', 'E01758980101', 'E01759000101', 'E01759000201', 'E01759020101', 'E01759030101', 'E01759040101', 'E01759070101', 'E01759080101', 'E01759090101', 'E01759090701', 'E01759110101', 'E01759110201', 'E01759130101', 'E01759130201', 'E01759130401', 'E01759180101', 'E01759280101', 'E01759300101', 'E01759340101', 'E01759350101', 'E01759350201', 'E01759360101', 'E01759370101', 'E01759380101', 'E01759400101', 'E01759410101', 'E01759420101', 'E01759440101', 'E01759450101', 'E01759460101', 'E01759490101', 'E01759500301', 'E01759500701', 'E01759510101', 'E01759510201', 'E01759510301', 'E01759510401', 'E01759520101', 'E01759560101', 'E01759580101', 'E01759630101', 'E01759660101', 'E01759730101', 'E01759800101', 'E01759800201', 'E01779100101', 'E01779130101', 'E01779140101', 'E01783260101', 'E01783690101', 'E01783720101', 'E01783740101', 'E01783750101', 'E01783780101', 'E01784800101', 'E01784800201', 'E01801820101', 'E01801830101', 'E01801850101', 'E01801860101', 'E01801870101', 'E01801900101', 'E01802920101', 'E01808500101', 'E01808510101', 'E01808520101', 'E01808530101', 'E01808540101', 'E01808550101', 'E01808550201', 'E01818110101', 'E01818160101', 'E01821420101', 'E01821490301', 'E01821590101', 'E01826330101', 'E01826350101', 'E01826370101', 'E01826390201', 'E01826390301', 'E01828300101', 'E01829610101', 'E01829650101', 'E01829670101', 'E01833600101', 'E01838880101', 'E01838890101', 'E01838890201', 'E01838900101', 'E01838910101', 'E01838940101', 'E01838950101', 'E01838960301', 'E01838960401', 'E01838960501', 'E01838960502', 'E01838970101', 'E01838980101', 'E01839000201', 'E01839010101', 'E01839050101', 'E01839050501', 'E01839060101', 'E01839070301', 'E01839100101', 'E01839120101', 'E01839160101', 'E01840140101', 'E01840150101', 'E01841440101', 'E01841460101', 'E01841490101', 'E01841660101', 'E01841670101', 'E01841690101', 'E01841720101', 'E01841740101', 'E01843160101', 'E01850080101', 'E01850100101', 'E01855000101', 'E01855020101', 'E01855080101', 'E01861050101', 'E01861060101', 'E01867790101', 'E01870510101', 'E01870550101', 'E01870590101', 'E01901190101', 'E01904800101', 'E01904830101', 'E01904850101', 'E01908520101', 'E01908540101', 'E01908560101', 'E01908580101', 'E01909920101', 'E01909960101', 'E01910000101', 'E01910010101', 'E01910010201', 'E01910010301', 'E01910010302', 'E01910010501', 'E01910090101', 'E01912560101', 'E01912560201', 'E01912600101', 'E01912610101', 'E01912630101', 'E01912790101', 'E01912810101', 'E01914200101', 'E01914210101', 'E01914930101', 'E01914950101', 'E01915440101', 'E01916230101', 'E01916250101', 'E01916260101', 'E01917310101', 'E01918650101', 'E01918670101', 'E01918720101', 'E01923360101', 'E01923380101', 'E01923390101', 'E01923420101', 'E01925150101', 'E01925160101', 'E01925610101', 'E01925620101', 'E01927010101', 'E01927500101', 'E01927950101', 'E01927970101', 'E01927990101', 'E01928000101', 'E01928000301', 'E01931810101', 'E01932200201', 'E01932200301', 'E01932230101', 'E01932760101', 'E01933380101', 'E01933420101', 'E01934530101', 'E01935430101', 'E01936180101', 'E01937880101', 'E01938000101', 'E01938030101', 'E01939170101', 'E01939560101', 'W00000010101', 'W00000020101', 'W00000060101', 'W00000060201', 'W00000060202', 'W00000060301', 'W00000060302', 'W00000070101', 'W00000080101', 'W00000090101', 'W00000100101', 'W00000120101', 'W00000130101', 'W00000140101', 'W00000150101', 'W00000160101', 'W00000170101', 'W00000210101', 'W00000220101', 'W00000230101', 'W00000260101', 'W00000290101', 'W00000310101', 'W00000320101', 'W00000430101', 'W00000440101', 'W00000470101', 'W00000480101', 'W00000500101', 'W00000520101', 'W00000530101', 'W00000570101', 'W00000620101', 'W00000630101', 'W00000660101', 'W00000680101', 'W00000690101', 'W00000700101', 'W00000710101', 'W00000720101', 'W00000730101', 'W00000750101', 'W00000780101', 'W00000790101', 'W00000810101', 'W00000860101', 'W00000870101', 'W00000920101', 'W00000950101', 'W00000980101', 'W00001010101', 'W00001030101', 'W00001040101', 'W00001100101', 'W00001110101', 'W00001120101', 'W00001130101', 'W00001140101', 'W00001150101', 'W00001160101', 'W00001160201', 'W00001170101', 'W00001240101', 'W00001250101', 'W00001260101', 'non_avoid_ip_post', 'avoid_ip_post', 'non_avoid_ed_post', 'avoid_ed_post', 'all_cause_ip_post', 'all_cause_ed_post', 'all_cause_acute_post', 'avoid_acute_post']

# COMMAND ----------

# Create a new dataframe with only the desired columns
df = df.select(*columns)
print(df.printSchema())

# COMMAND ----------

# # Assume 'df' is the original DataFrame with 18M rows

# # Calculate the fraction to sample in order to get approximately 500k rows
# fraction = 25 / df.count()

# # Take a random sample from the DataFrame
# sampled_df = df.sample(withReplacement=False, fraction=fraction, seed=42)

# # Show the number of rows in the sampled DataFrame
# print("Number of rows in the sampled DataFrame:", sampled_df.count())

# df = sampled_df
# print((df.count(), len(df.columns)))

# COMMAND ----------

# # Assuming pred1 is your PySpark DataFrame containing categorical features
# categorical_columns = []  # List to store the categorical feature column names

# # Loop through the columns of pred1
# for column in df.columns[2:]:
#     categorical_columns.append(column)

# # Print the categorical feature column names
# print(categorical_columns)

# COMMAND ----------

categorical_columns = ['ageCat', 'sex']

numerical_columns = ['BLD001', 'BLD002', 'BLD003', 'BLD004', 'BLD005', 'BLD006', 'BLD007', 'BLD008', 'BLD009', 'BLD010', 'CIR001', 'CIR002', 'CIR003', 'CIR004', 'CIR005', 'CIR006', 'CIR007', 'CIR008', 'CIR009', 'CIR010', 'CIR011', 'CIR012', 'CIR013', 'CIR014', 'CIR015', 'CIR016', 'CIR017', 'CIR018', 'CIR019', 'CIR020', 'CIR021', 'CIR022', 'CIR023', 'CIR024', 'CIR025', 'CIR026', 'CIR027', 'CIR028', 'CIR029', 'CIR030', 'CIR031', 'CIR032', 'CIR033', 'CIR034', 'CIR035', 'CIR036', 'CIR037', 'CIR038', 'CIR039', 'DEN001', 'DIG001', 'DIG004', 'DIG005', 'DIG006', 'DIG007', 'DIG008', 'DIG009', 'DIG010', 'DIG011', 'DIG012', 'DIG013', 'DIG014', 'DIG015', 'DIG016', 'DIG017', 'DIG018', 'DIG019', 'DIG020', 'DIG021', 'DIG022', 'DIG023', 'DIG024', 'DIG025', 'EAR001', 'EAR002', 'EAR003', 'EAR004', 'EAR005', 'EAR006', 'END001', 'END002', 'END003', 'END007', 'END008', 'END009', 'END010', 'END011', 'END012', 'END013', 'END014', 'END015', 'END016', 'END017', 'EXT001', 'EXT002', 'EXT003', 'EXT004', 'EXT005', 'EXT006', 'EXT007', 'EXT008', 'EXT009', 'EXT010', 'EXT011', 'EXT012', 'EXT013', 'EXT014', 'EXT015', 'EXT016', 'EXT017', 'EXT018', 'EXT019', 'EXT025', 'EXT026', 'EXT027', 'EXT028', 'EXT029', 'EXT030', 'EYE001', 'EYE002', 'EYE003', 'EYE004', 'EYE005', 'EYE006', 'EYE007', 'EYE008', 'EYE009', 'EYE010', 'EYE011', 'EYE012', 'FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005', 'FAC006', 'FAC007', 'FAC008', 'FAC009', 'FAC010', 'FAC011', 'FAC012', 'FAC013', 'FAC014', 'FAC015', 'FAC016', 'FAC017', 'FAC018', 'FAC019', 'FAC020', 'FAC021', 'FAC022', 'FAC023', 'FAC024', 'FAC025', 'GEN001', 'GEN002', 'GEN003', 'GEN004', 'GEN005', 'GEN006', 'GEN007', 'GEN008', 'GEN009', 'GEN010', 'GEN011', 'GEN012', 'GEN013', 'GEN014', 'GEN015', 'GEN016', 'GEN017', 'GEN018', 'GEN019', 'GEN020', 'GEN021', 'GEN022', 'GEN023', 'GEN024', 'GEN025', 'GEN026', 'INF001', 'INF002', 'INF003', 'INF004', 'INF005', 'INF006', 'INF007', 'INF008', 'INF009', 'INF010', 'INF011', 'INJ001', 'INJ002', 'INJ003', 'INJ004', 'INJ005', 'INJ006', 'INJ007', 'INJ008', 'INJ009', 'INJ010', 'INJ011', 'INJ012', 'INJ013', 'INJ014', 'INJ015', 'INJ016', 'INJ017', 'INJ018', 'INJ019', 'INJ021', 'INJ024', 'INJ025', 'INJ026', 'INJ027', 'INJ028', 'INJ029', 'INJ030', 'INJ031', 'INJ032', 'INJ033', 'INJ034', 'INJ035', 'INJ036', 'INJ037', 'INJ038', 'INJ039', 'INJ040', 'INJ041', 'INJ042', 'INJ043', 'INJ044', 'INJ045', 'INJ046', 'INJ047', 'INJ048', 'INJ049', 'INJ050', 'INJ051', 'INJ052', 'INJ053', 'INJ054', 'INJ055', 'INJ056', 'INJ057', 'INJ058', 'INJ059', 'INJ060', 'INJ061', 'INJ062', 'INJ063', 'INJ064', 'INJ065', 'INJ066', 'INJ067', 'INJ068', 'INJ069', 'INJ070', 'INJ071', 'INJ072', 'INJ073', 'INJ074', 'INJ075', 'INJ076', 'MAL001', 'MAL002', 'MAL003', 'MAL004', 'MAL005', 'MAL006', 'MAL007', 'MAL008', 'MAL009', 'MAL010', 'MBD001', 'MBD002', 'MBD003', 'MBD004', 'MBD005', 'MBD006', 'MBD007', 'MBD008', 'MBD009', 'MBD010', 'MBD011', 'MBD012', 'MBD013', 'MBD014', 'MBD017', 'MBD018', 'MBD019', 'MBD020', 'MBD021', 'MBD022', 'MBD023', 'MBD024', 'MBD025', 'MBD026', 'MUS001', 'MUS002', 'MUS003', 'MUS004', 'MUS005', 'MUS006', 'MUS007', 'MUS008', 'MUS009', 'MUS010', 'MUS011', 'MUS012', 'MUS013', 'MUS014', 'MUS015', 'MUS016', 'MUS017', 'MUS018', 'MUS019', 'MUS020', 'MUS021', 'MUS022', 'MUS023', 'MUS024', 'MUS025', 'MUS026', 'MUS028', 'MUS030', 'MUS031', 'MUS032', 'MUS033', 'MUS034', 'MUS036', 'MUS037', 'MUS038', 'NEO001', 'NEO002', 'NEO003', 'NEO004', 'NEO005', 'NEO006', 'NEO007', 'NEO008', 'NEO009', 'NEO010', 'NEO011', 'NEO012', 'NEO013', 'NEO014', 'NEO015', 'NEO016', 'NEO017', 'NEO018', 'NEO019', 'NEO020', 'NEO021', 'NEO022', 'NEO023', 'NEO024', 'NEO025', 'NEO026', 'NEO027', 'NEO028', 'NEO029', 'NEO030', 'NEO031', 'NEO032', 'NEO033', 'NEO034', 'NEO035', 'NEO036', 'NEO037', 'NEO038', 'NEO039', 'NEO040', 'NEO041', 'NEO042', 'NEO043', 'NEO044', 'NEO045', 'NEO046', 'NEO047', 'NEO048', 'NEO049', 'NEO050', 'NEO051', 'NEO052', 'NEO053', 'NEO054', 'NEO055', 'NEO056', 'NEO057', 'NEO058', 'NEO059', 'NEO060', 'NEO061', 'NEO062', 'NEO063', 'NEO064', 'NEO065', 'NEO066', 'NEO067', 'NEO068', 'NEO069', 'NEO070', 'NEO071', 'NEO072', 'NEO073', 'NEO074', 'NVS001', 'NVS002', 'NVS003', 'NVS004', 'NVS005', 'NVS006', 'NVS007', 'NVS008', 'NVS009', 'NVS010', 'NVS011', 'NVS012', 'NVS013', 'NVS014', 'NVS015', 'NVS016', 'NVS017', 'NVS018', 'NVS019', 'NVS020', 'NVS021', 'NVS022', 'PNL001', 'PNL002', 'PNL003', 'PNL004', 'PNL005', 'PNL006', 'PNL007', 'PNL008', 'PNL009', 'PNL010', 'PNL011', 'PNL012', 'PNL013', 'PNL014', 'PRG001', 'PRG002', 'PRG003', 'PRG004', 'PRG005', 'PRG006', 'PRG007', 'PRG008', 'PRG009', 'PRG010', 'PRG011', 'PRG012', 'PRG013', 'PRG014', 'PRG015', 'PRG016', 'PRG017', 'PRG018', 'PRG020', 'PRG021', 'PRG022', 'PRG023', 'PRG024', 'PRG025', 'PRG026', 'PRG027', 'PRG028', 'PRG029', 'PRG030', 'RSP001', 'RSP002', 'RSP003', 'RSP004', 'RSP005', 'RSP006', 'RSP007', 'RSP008', 'RSP009', 'RSP010', 'RSP011', 'RSP012', 'RSP013', 'RSP014', 'RSP015', 'RSP016', 'RSP017', 'SKN001', 'SKN002', 'SKN003', 'SKN004', 'SKN005', 'SKN006', 'SKN007', 'SYM001', 'SYM002', 'SYM003', 'SYM004', 'SYM005', 'SYM006', 'SYM007', 'SYM008', 'SYM009', 'SYM010', 'SYM011', 'SYM012', 'SYM013', 'SYM014', 'SYM015', 'SYM016', 'SYM017', 'E01754130101', 'E01754140101', 'E01754150101', 'E01754160101', 'E01754180101', 'E01754180201', 'E01754190101', 'E01754190201', 'E01754200101', 'E01754200201', 'E01754210101', 'E01754210201', 'E01754230101', 'E01754260101', 'E01754260401', 'E01754270101', 'E01754280101', 'E01754290101', 'E01754300101', 'E01754300201', 'E01754310101', 'E01754340101', 'E01754350101', 'E01754350201', 'E01754430101', 'E01754430201', 'E01754530101', 'E01754540101', 'E01754570101', 'E01754610101', 'E01754620101', 'E01754630101', 'E01754630201', 'E01754630202', 'E01754660101', 'E01754660301', 'E01754670101', 'E01754680101', 'E01754700101', 'E01754730101', 'E01754760101', 'E01754770101', 'E01754770201', 'E01754770202', 'E01754770301', 'E01754810101', 'E01754820101', 'E01754820201', 'E01754830101', 'E01754830201', 'E01754840101', 'E01754850101', 'E01754860101', 'E01754870101', 'E01754870201', 'E01754880101', 'E01754880201', 'E01754890101', 'E01754890201', 'E01754890202', 'E01754910101', 'E01754930101', 'E01754940101', 'E01754950101', 'E01754960101', 'E01754960201', 'E01754970101', 'E01754970201', 'E01754980101', 'E01754990101', 'E01755000101', 'E01755010101', 'E01755030101', 'E01755050101', 'E01755070101', 'E01755090101', 'E01755110101', 'E01755140101', 'E01755150101', 'E01755170101', 'E01755170201', 'E01755190101', 'E01755210101', 'E01755220101', 'E01755240101', 'E01755250101', 'E01755350101', 'E01755390101', 'E01755430101', 'E01755520101', 'E01755530101', 'E01755540101', 'E01755550101', 'E01755560101', 'E01755570101', 'E01755570201', 'E01755580101', 'E01755590101', 'E01755600101', 'E01755610101', 'E01755610201', 'E01755610202', 'E01755610203', 'E01755610301', 'E01755620101', 'E01755620201', 'E01755620202', 'E01755620203', 'E01755630101', 'E01755630201', 'E01755640101', 'E01755640201', 'E01755650101', 'E01755650201', 'E01755650202', 'E01755650203', 'E01755650401', 'E01755660101', 'E01755680101', 'E01755700101', 'E01755720101', 'E01755730101', 'E01755740101', 'E01755740201', 'E01755740202', 'E01755740203', 'E01755740301', 'E01755740303', 'E01755750101', 'E01755760101', 'E01755760201', 'E01755760202', 'E01755760203', 'E01755760204', 'E01755760206', 'E01755760207', 'E01755760209', 'E01755760302', 'E01755760401', 'E01755780101', 'E01755790101', 'E01755790201', 'E01755800101', 'E01755810101', 'E01755820101', 'E01755830101', 'E01755840101', 'E01755860101', 'E01755870101', 'E01755870201', 'E01755870202', 'E01755880101', 'E01755890101', 'E01755900101', 'E01755910101', 'E01755920101', 'E01755940101', 'E01755940401', 'E01755940501', 'E01755950101', 'E01755960101', 'E01755960401', 'E01755970101', 'E01755980101', 'E01755990101', 'E01756010101', 'E01756020101', 'E01756030101', 'E01756040101', 'E01756050101', 'E01756060101', 'E01756070101', 'E01756080101', 'E01756090101', 'E01756100101', 'E01756110101', 'E01756120101', 'E01756130101', 'E01756160101', 'E01756230101', 'E01756250101', 'E01756300101', 'E01756340101', 'E01756370101', 'E01756380101', 'E01756550101', 'E01756550201', 'E01756560101', 'E01756570101', 'E01756570201', 'E01756610101', 'E01756640101', 'E01756650101', 'E01756660101', 'E01756670101', 'E01756690101', 'E01756790101', 'E01756800101', 'E01756810101', 'E01756820101', 'E01756820201', 'E01756820301', 'E01756820302', 'E01756820501', 'E01756830101', 'E01756890101', 'E01756900101', 'E01756900201', 'E01756910101', 'E01756910201', 'E01756910202', 'E01756920101', 'E01756930101', 'E01756930501', 'E01756940101', 'E01756940201', 'E01756950101', 'E01756960101', 'E01756980101', 'E01757000101', 'E01757000201', 'E01757050101', 'E01757060101', 'E01757100101', 'E01757120101', 'E01757130101', 'E01757160101', 'E01757190101', 'E01757200101', 'E01757220101', 'E01757220201', 'E01757220202', 'E01757220203', 'E01757220204', 'E01757220205', 'E01757220206', 'E01757220207', 'E01757230101', 'E01757230201', 'E01757370101', 'E01757370301', 'E01757380101', 'E01757390101', 'E01757390301', 'E01757400101', 'E01757430101', 'E01757440101', 'E01757450101', 'E01757460101', 'E01757460201', 'E01757460202', 'E01757460301', 'E01757490101', 'E01757500101', 'E01757500201', 'E01757510101', 'E01757520101', 'E01757530101', 'E01757530201', 'E01757540101', 'E01757570101', 'E01757580101', 'E01757590101', 'E01757650101', 'E01757660101', 'E01757680101', 'E01757690101', 'E01757710101', 'E01757750101', 'E01757760101', 'E01757770101', 'E01757790101', 'E01757800101', 'E01757810101', 'E01757820101', 'E01757830101', 'E01757840101', 'E01757840201', 'E01757850101', 'E01757860101', 'E01757900101', 'E01757940101', 'E01757960101', 'E01758000101', 'E01758010101', 'E01758020101', 'E01758050101', 'E01758070101', 'E01758090101', 'E01758100101', 'E01758110101', 'E01758110201', 'E01758120101', 'E01758140101', 'E01758150101', 'E01758180101', 'E01758180201', 'E01758190101', 'E01758200101', 'E01758210101', 'E01758220101', 'E01758230101', 'E01758240101', 'E01758250101', 'E01758250201', 'E01758250203', 'E01758250302', 'E01758260101', 'E01758270101', 'E01758280101', 'E01758310101', 'E01758350101', 'E01758360101', 'E01758360201', 'E01758370101', 'E01758380101', 'E01758390101', 'E01758400101', 'E01758410101', 'E01758420101', 'E01758430101', 'E01758450101', 'E01758470101', 'E01758480101', 'E01758490101', 'E01758540101', 'E01758740101', 'E01758750101', 'E01758770101', 'E01758770201', 'E01758790101', 'E01758810101', 'E01758820101', 'E01758840101', 'E01758850101', 'E01758870101', 'E01758870201', 'E01758890101', 'E01758950101', 'E01758980101', 'E01759000101', 'E01759000201', 'E01759020101', 'E01759030101', 'E01759040101', 'E01759070101', 'E01759080101', 'E01759090101', 'E01759090701', 'E01759110101', 'E01759110201', 'E01759130101', 'E01759130201', 'E01759130401', 'E01759180101', 'E01759280101', 'E01759300101', 'E01759340101', 'E01759350101', 'E01759350201', 'E01759360101', 'E01759370101', 'E01759380101', 'E01759400101', 'E01759410101', 'E01759420101', 'E01759440101', 'E01759450101', 'E01759460101', 'E01759490101', 'E01759500301', 'E01759500701', 'E01759510101', 'E01759510201', 'E01759510301', 'E01759510401', 'E01759520101', 'E01759560101', 'E01759580101', 'E01759630101', 'E01759660101', 'E01759730101', 'E01759800101', 'E01759800201', 'E01779100101', 'E01779130101', 'E01779140101', 'E01783260101', 'E01783690101', 'E01783720101', 'E01783740101', 'E01783750101', 'E01783780101', 'E01784800101', 'E01784800201', 'E01801820101', 'E01801830101', 'E01801850101', 'E01801860101', 'E01801870101', 'E01801900101', 'E01802920101', 'E01808500101', 'E01808510101', 'E01808520101', 'E01808530101', 'E01808540101', 'E01808550101', 'E01808550201', 'E01818110101', 'E01818160101', 'E01821420101', 'E01821490301', 'E01821590101', 'E01826330101', 'E01826350101', 'E01826370101', 'E01826390201', 'E01826390301', 'E01828300101', 'E01829610101', 'E01829650101', 'E01829670101', 'E01833600101', 'E01838880101', 'E01838890101', 'E01838890201', 'E01838900101', 'E01838910101', 'E01838940101', 'E01838950101', 'E01838960301', 'E01838960401', 'E01838960501', 'E01838960502', 'E01838970101', 'E01838980101', 'E01839000201', 'E01839010101', 'E01839050101', 'E01839050501', 'E01839060101', 'E01839070301', 'E01839100101', 'E01839120101', 'E01839160101', 'E01840140101', 'E01840150101', 'E01841440101', 'E01841460101', 'E01841490101', 'E01841660101', 'E01841670101', 'E01841690101', 'E01841720101', 'E01841740101', 'E01843160101', 'E01850080101', 'E01850100101', 'E01855000101', 'E01855020101', 'E01855080101', 'E01861050101', 'E01861060101', 'E01867790101', 'E01870510101', 'E01870550101', 'E01870590101', 'E01901190101', 'E01904800101', 'E01904830101', 'E01904850101', 'E01908520101', 'E01908540101', 'E01908560101', 'E01908580101', 'E01909920101', 'E01909960101', 'E01910000101', 'E01910010101', 'E01910010201', 'E01910010301', 'E01910010302', 'E01910010501', 'E01910090101', 'E01912560101', 'E01912560201', 'E01912600101', 'E01912610101', 'E01912630101', 'E01912790101', 'E01912810101', 'E01914200101', 'E01914210101', 'E01914930101', 'E01914950101', 'E01915440101', 'E01916230101', 'E01916250101', 'E01916260101', 'E01917310101', 'E01918650101', 'E01918670101', 'E01918720101', 'E01923360101', 'E01923380101', 'E01923390101', 'E01923420101', 'E01925150101', 'E01925160101', 'E01925610101', 'E01925620101', 'E01927010101', 'E01927500101', 'E01927950101', 'E01927970101', 'E01927990101', 'E01928000101', 'E01928000301', 'E01931810101', 'E01932200201', 'E01932200301', 'E01932230101', 'E01932760101', 'E01933380101', 'E01933420101', 'E01934530101', 'E01935430101', 'E01936180101', 'E01937880101', 'E01938000101', 'E01938030101', 'E01939170101', 'E01939560101', 'W00000010101', 'W00000020101', 'W00000060101', 'W00000060201', 'W00000060202', 'W00000060301', 'W00000060302', 'W00000070101', 'W00000080101', 'W00000090101', 'W00000100101', 'W00000120101', 'W00000130101', 'W00000140101', 'W00000150101', 'W00000160101', 'W00000170101', 'W00000210101', 'W00000220101', 'W00000230101', 'W00000260101', 'W00000290101', 'W00000310101', 'W00000320101', 'W00000430101', 'W00000440101', 'W00000470101', 'W00000480101', 'W00000500101', 'W00000520101', 'W00000530101', 'W00000570101', 'W00000620101', 'W00000630101', 'W00000660101', 'W00000680101', 'W00000690101', 'W00000700101', 'W00000710101', 'W00000720101', 'W00000730101', 'W00000750101', 'W00000780101', 'W00000790101', 'W00000810101', 'W00000860101', 'W00000870101', 'W00000920101', 'W00000950101', 'W00000980101', 'W00001010101', 'W00001030101', 'W00001040101', 'W00001100101', 'W00001110101', 'W00001120101', 'W00001130101', 'W00001140101', 'W00001150101', 'W00001160101', 'W00001160201', 'W00001170101', 'W00001240101', 'W00001250101', 'W00001260101']

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Categorical Features (which are all strings)
stringindexers = [StringIndexer(
    inputCol=column, 
    outputCol=column+"_Index",
    handleInvalid="keep") for column in categorical_columns]

onehotencoders = [OneHotEncoder(
    inputCol=column+"_Index",
    outputCol=column+"_Vec") for column in categorical_columns]

categorical_columns_class_vector = [col + "_Vec" for col in categorical_columns]

# Assembler for categorical columns
assembler_categorical = VectorAssembler(
    inputCols=categorical_columns_class_vector,
    outputCol="categorical_features")

# Assembler for numeric columns
assembler_numeric = VectorAssembler(
    inputCols=numerical_columns,
    outputCol="numeric_features")

# Combine categorical and numeric features
assembler_combined = VectorAssembler(
    inputCols=["categorical_features", "numeric_features"],
    outputCol="features")

# Create the pipeline with all the stages
pipeline = Pipeline(
    stages=[*stringindexers,
            *onehotencoders,
            assembler_categorical,
            assembler_numeric,
            assembler_combined])

# Fit and transform the data with the pipeline
transformed_data = pipeline.fit(df).transform(df)

# COMMAND ----------

print(transformed_data.printSchema())

# COMMAND ----------

display(transformed_data)

# COMMAND ----------

transformed_data = transformed_data.select("beneID","state","features","all_cause_acute_post","avoid_acute_post")

# COMMAND ----------

transformed_data.write.saveAsTable("dua_058828_spa240.stage2_cost_vector_baseline_variables", mode='overwrite')

# COMMAND ----------

