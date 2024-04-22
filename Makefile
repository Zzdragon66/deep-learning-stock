.PHONY: SCRAPE_DATA
include .env


ScrapeDataDir := DataScrape
SPARK := spark 
DB := postgres

DB_USER := stock 
DB_PWD := stock 
DB_NAME := stock

INIT_DB:


SCRAPE_DATA:
