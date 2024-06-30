.PHONY: SCRAPE_DATA


ScrapeDataDir := DataScrape
SPARK := spark 
DB := postgres

DB_USER := stock 
DB_PWD := stock 
DB_NAME := stock

INIT_DB:
	echo $(DB_USER)

SCRAPE_DATA:
