import json
# from ai_ta_backend.model.response import FunctionMessage, ToolInvocation
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import (
		ChatPromptTemplate,
		FewShotPromptTemplate,
		MessagesPlaceholder,
		PromptTemplate,
		SystemMessagePromptTemplate,
)
import os
import logging
from flask_sqlalchemy import SQLAlchemy

from langchain_openai import ChatOpenAI

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import random
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain.tools.render import format_tool_to_openai_function


from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

from langchain_core.agents import AgentFinish
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings



def get_dynamic_prompt_template():
		
		examples = [
	{
        "input": "How many names were authored by Roxb?",
        "query": 'SELECT COUNT(*) as unique_pairs_count FROM (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "authorName" LIKE "%Roxb%" AND "recordType" IN ("AN", "SN"));'
    },
    {
        "input": "How many species have distributions in Myanmar, Meghalaya and Andhra Pradesh?",
        "query": 'SELECT COUNT(*) FROM (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Myanmar%" AND additionalDetail2 LIKE "%Meghalaya%" AND "additionalDetail2" LIKE "%Andhra Pradesh%"));'
    },
    {
        "input": "List the accepted names common to Myanmar, Meghalaya, Odisha, Andhra Pradesh.",
        "query": 'SELECT DISTINCT scientificName FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Myanmar%" AND additionalDetail2 LIKE "%Meghalaya%" AND "additionalDetail2" LIKE "%Odisha%" AND "additionalDetail2" LIKE "%Andhra Pradesh%"));'
    },
    {
        "input": "List the accepted names that represent 'tree'.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "HB" AND "additionalDetail2" LIKE "%tree%");'
    },
    {
        "input": "List the accepted names linked with Endemic tag.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Endemic%");'
    },
    {
        "input": "Name some plants which are tagged as endemic.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Endemic%");'

    },
    {
        "input": "List the accepted names published in Fl. Brit. India [J. D. Hooker].",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("publication" LIKE "%Fl. Brit. India [J. D. Hooker]%" OR "Publication" LIKE "%[J. D. Hooker]%" OR "Publication" LIKE "%Fl. Brit. India%");'
    },
    {
        "input": "Name some plants published in Bot. Zeitung.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "publication" LIKE "%Bot. Zeitung%";'
    },
    {
        "input": "List some species published in Acta Horti Gothub. after 1930.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "publication" LIKE "%Acta Horti Gothub%" AND "yearOfpublication" > 1930;'
    },
    {
        "input": "How many plants do not have a designated type?",
        "query": 'SELECT COUNT(DISTINCT "scientificName") FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "TY" AND "additionalDetail2" LIKE "%not designated%");'
    },
    {
        "input": "How many accepted names have ‘Silhet’/ ‘Sylhet’ in their Type?",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "TY" AND ("additionalDetail2" LIKE "%Silhet%" OR "additionalDetail2" LIKE "%Sylhet%"));'
    },
    {
        "input": "How many plants or species have ‘Silhet’/ ‘Sylhet’ in their Type?",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "TY" AND ("additionalDetail2" LIKE "%Silhet%" OR "additionalDetail2" LIKE "%Sylhet%"));'
    },
    {
        "input": "How many species were distributed in Sikkim and Meghalaya?",
        "query": 'SELECT COUNT(*) AS unique_pairs FROM (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Sikkim%" AND additionalDetail2 LIKE "%Meghalaya%"));'
    },
    {
        "input": "List the accepted names common to Kerala, Tamil Nadu, Andhra Pradesh, Karnataka, Maharashtra, Odisha, Meghalaya and Myanmar.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Myanmar%" AND additionalDetail2 LIKE "%Meghalaya%" AND "additionalDetail2" LIKE "%Odisha%" AND "additionalDetail2" LIKE "%Andhra Pradesh%" AND "additionalDetail2" LIKE "%Kerala%" AND "additionalDetail2" LIKE "%Tamil Nadu%" AND "additionalDetail2" LIKE "%Karnataka%" AND "additionalDetail2" LIKE "%Maharashtra%"));'
    },
    {
        "input": "List the accepted names common to Europe, Afghanistan, Jammu & Kashmir, Himachal, Nepal, Sikkim, Bhutan, Arunachal Pradesh and China.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Europe%" AND additionalDetail2 LIKE "%Afghanistan%" AND "additionalDetail2" LIKE "%Jammu & Kashmir%" AND "additionalDetail2" LIKE "%Himachal%" AND "additionalDetail2" LIKE "%Nepal%" AND "additionalDetail2" LIKE "%Sikkim%" AND "additionalDetail2" LIKE "%Bhutan%" AND "additionalDetail2" LIKE "%Arunachal Pradesh%" AND "additionalDetail2" LIKE "%China%"));'
    },
    {
        "input": "List the accepted names common to Europe, Afghanistan, Austria, Belgium, Czechoslovakia, Denmark, France, Greece, Hungary, Italy, Moldava, Netherlands, Poland, Romania, Spain, Switzerland, Jammu & Kashmir, Himachal, Nepal, and China.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Europe%" AND additionalDetail2 LIKE "%Afghanistan%" AND "additionalDetail2" LIKE "%Jammu & Kashmir%" AND "additionalDetail2" LIKE "%Himachal%" AND "additionalDetail2" LIKE "%Nepal%" AND "additionalDetail2" LIKE "%Austria%" AND "additionalDetail2" LIKE "%Belgium%" AND "additionalDetail2" LIKE "%Czechoslovakia%" AND "additionalDetail2" LIKE "%China%" AND "additionalDetail2" LIKE "%Denmark%" AND "additionalDetail2" LIKE "%Greece%" AND "additionalDetail2" LIKE "%France%" AND "additionalDetail2" LIKE "%Hungary%" AND "additionalDetail2" LIKE "%Italy%" AND "additionalDetail2" LIKE "%Moldava%" AND "additionalDetail2" LIKE "%Netherlands%" AND "additionalDetail2" LIKE "%Poland%" AND "additionalDetail2" LIKE "%Poland%" AND "additionalDetail2" LIKE "%Romania%" AND "additionalDetail2" LIKE "%Spain%" AND "additionalDetail2" LIKE "%Switzerland%"));'
    },
    {
        "input": "List the species which are distributed in Sikkim and Meghalaya.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Sikkim%" AND additionalDetail2 LIKE "%Meghalaya%"));'
    },
    {
        "input": "How many species are common to America, Europe, Africa, Asia, and Australia?",
        "query": 'SELECT COUNT(*) AS unique_pairs IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%America%" AND additionalDetail2 LIKE "%Europe%" AND "additionalDetail2" LIKE "%Africa%" AND "additionalDetail2" LIKE "%Asia%" AND "additionalDetail2" LIKE "%Australia%"));'
    },
    {
        "input": "List the species names common to India and Myanmar, Malaysia, Indonesia, and Australia.",
        "query": 'SELECT DISTINCT "Scientific_Name" FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%India%" AND additionalDetail2 LIKE "%Myanmar%" AND additionalDetail2 LIKE "%Malaysia%" AND additionalDetail2 LIKE "%Indonesia%" AND additionalDetail2 LIKE "%Australia%"));'
    },
    {
        "input": "List all plants which are tagged as urban.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "urban" = "YES";'
    },
    {
        "input": "List all plants which are tagged as fruit.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "fruit" = "YES";'
    },
    {
        "input": "List all plants which are tagged as medicinal.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "medicinal" = "YES";'
    },
    {
        "input": "List all family names which are gymnosperms.",
        "query": 'SELECT DISTINCT "familyName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "group" = "Gymnosperms";'
    },
    {
        "input": "How many accepted names are tagged as angiosperms?",
        "query": 'SELECT COUNT(DISTINCT "scientificName") FROM plants WHERE "recordType" = "AN" AND "group" = "Angiosperms";'
    },
    {
        "input": "How many accepted names belong to the 'Saxifraga' genus?",
        "query": 'SELECT COUNT(DISTINCT "scientificName") FROM plants WHERE "genusName" = "Saxifraga" AND "recordType" = "AN";'
    },
    {
        "input": "List the accepted names tagged as 'perennial herb' or 'climber'.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "HB" AND (("additionalDetail2" LIKE "%perennial%" AND "additionalDetail2" LIKE "%herb%") OR "additionalDetail2" LIKE "%climber%")));'
    },
    {
        "input": "List the species that represent aquatic herb.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "HB" AND (("additionalDetail2" LIKE "%aquatic%" AND "additionalDetail2" LIKE "%herb%") OR "additionalDetail2" LIKE "%climber%")));'
    },
    {
        "input": "How many plants are represented as a succulent chamerophyte?",
        "query": 'SELECT COUNT(DISTINCT "scientificName") FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "HB" AND "additionalDetail2" LIKE "%succulent%" AND "additionalDetail2" LIKE "%chamerophyte%");'

    },
    {
        "input": "How many accepted names are native to South Africa?",
        "query": 'SELECT COUNT(DISTINCT scientificName) FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%native%" AND "additionalDetail2" LIKE "%south%" AND "additionalDetail2" LIKE "%africa%");'

    },
    {
        "input": "How many accepted names are native to China?",
        "query": 'SELECT COUNT(DISTINCT scientificName) FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%native%" AND "additionalDetail2" LIKE "%china%");'

    },
    {
        "input": "List the accepted names which were introduced and naturalized.",
        "query": 'SELECT DISTINCT "scientificName FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%introduced%" AND "additionalDetail2" LIKE "%naturalized%");'
    },
    {
        "input": "How many accepted names are cultivated?",
        "query": 'SELECT COUNT(DISTINCT "scientificName) FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%cultivated%");'

    },
    {
        "input": "List all ornamental plants.",
        "query": 'SELECT DISTINCT "scientificName FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%ornamental%");'
    },
    {
        "input": "How many plants from the 'Leguminosae' family have a altitudinal range up to 1000 m?",
        "query": 'SELECT COUNT(*) FROM plants WHERE "recordType" = "AL" AND "familyName" = "Leguminosae" AND "additionalDetail2" LIKE "%1000%";'
    },
    {
        "input": "List the accepted names linked with the 'endemic' tag for Karnataka.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Endemic%" AND "additionalDetail2" LIKE "%Karnataka%");'
    },
    {
        "input": "How many species did Vahl discover?",
        "query": 'SELECT COUNT(DISTINCT "scientificName") FROM plants WHERE "recordType" IN ("AN", "SN") AND "authorName" LIKE "%Vahl%";'
    },
    {
        "input": "How many names are rooted with basionym?",
        "query": 'SELECT COUNT(DISTINCT "scientificName") FROM plants WHERE "recordType" IN ("AN", "SN") AND "authorName" LIKE "%(%;',
    },
    {
      "input": "How many plants are introduced and naturalized?",
      "query": 'SELECT COUNT(DISTINCT "scientificName") FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%introduced%" AND ("additionalDetail2" LIKE "%naturalized%" OR "additionalDetail2" LIKE "%naturalised%"));',
    },
    {
        "input": "List some endangered plant species.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%endangered%");',
    },
    {
        "input": "Which plants are represented as biennial and perennial herb?",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "HB" AND "additionalDetail2" LIKE "%herb%" AND ("additionalDetail2" LIKE "%biennial%" OR "additionalDetail2" LIKE "%perennial%"));',
    },
    {
        "input": "Which genera have basionyms?",
        "query": 'SELECT DISTINCT genusName FROM plants WHERE "recordType" IN ("AN", "SN") AND "authorName" LIKE "%(%;',
    },
    {
        "input": "List some basionyms found by Aiton.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "authorName" LIKE "%Aiton%" AND "authorName" LIKE "%(%");',
    },
    {
        "input": "List the plant species belonging to the genus 'Mangifera' found in the state of Maharashtra.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "genusName" LIKE "%Mangifera%" AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Maharashtra%");',
    },
    {
        "input": "Find all plant species in the family 'Rosaceae' found in the United States.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "familyName" LIKE "%Rosaceae%" AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%United States%");',
    },
    {
        "input": "Find the species of the genus 'Pinus' located in Europe.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "genusName" LIKE "%Pinus%" AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Europe%");',
    },
    {
        "input": "List the plants from the 'Myrtaceae' family that are found in South America.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "familyName" LIKE "%Myrtaceae%" AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%South America%");',
    },
    {
        "input": "Find all plant species in the family 'Rubiaceae' found in Africa.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "familyName" LIKE "%Rubiaceae%" AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Africa%");',
    },
    {
        "input": "List the species of the genus 'Lavandula' in the family 'Lamiaceae' that are native to the Mediterranean region.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "familyName" LIKE "%Lamiaceae%" AND "genusName" LIKE "%Lavandula%" AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "RE" AND "additionalDetail2" LIKE "%native%" AND "additionalDetail2" LIKE "%Mediterranean%");',
    },
    {
        "input": "Find the genera of plants in the 'Asteraceae' family that are found in both China and Japan.",
        "query": 'SELECT DISTINCT "genusName" FROM plants WHERE "recordType" = "GE" AND "familyName" LIKE "%Asteraceae%" AND ("familyNumber", "genusNumber") IN (SELECT "familyNumber", "genusNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%China%" AND "additionalDetail2" LIKE "%Japan%");'
    },
    {
        "input": "List the species of the genus 'Acer' found in Canada.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "genusName" LIKE "%Acer%" AND ("familyNumber", "genusNumber") IN (SELECT "familyNumber", "genusNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Canada%");'
    },
    {
        "input": "List the basionyms authored by L.",
        "query": 'SELECT DISTINCT "scientificName" FROM plants WHERE "recordType" IN ("AN", "SN") AND "authorName" LIKE "%L.%" AND "authorName" LIKE "%(%";',
    },
    {
        "input": "List the genera of plants in the family 'Fabaceae' found in both India and Australia.",
        "query": """SELECT DISTINCT "genusName" FROM plants WHERE "recordType" = "GE" AND "familyName" LIKE "%Fabaceae%" AND
        ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND
        "additionalDetail2" LIKE "%Australia%" AND
        ("additionalDetail2" LIKE "%India%"
        OR ("additionalDetail2" LIKE "%Andhra Pradesh%")
   OR ("additionalDetail2" LIKE "%Arunachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Assam%")
   OR ("additionalDetail2" LIKE "%Bihar%")
   OR ("additionalDetail2" LIKE "%Chhattisgarh%")
   OR ("additionalDetail2" LIKE "%Goa%")
   OR ("additionalDetail2" LIKE "%Gujarat%")
   OR ("additionalDetail2" LIKE "%Haryana%")
   OR ("additionalDetail2" LIKE "%Himachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Jharkhand%")
   OR ("additionalDetail2" LIKE "%Karnataka%")
   OR ("additionalDetail2" LIKE "%Kerala%")
   OR ("additionalDetail2" LIKE "%Madhya Pradesh%")
   OR ("additionalDetail2" LIKE "%Maharashtra%")
   OR ("additionalDetail2" LIKE "%Manipur%")
   OR ("additionalDetail2" LIKE "%Meghalaya%")
   OR ("additionalDetail2" LIKE "%Mizoram%")
   OR ("additionalDetail2" LIKE "%Nagaland%")
   OR ("additionalDetail2" LIKE "%Odisha%")
   OR ("additionalDetail2" LIKE "%Punjab%")
   OR ("additionalDetail2" LIKE "%Rajasthan%")
   OR ("additionalDetail2" LIKE "%Sikkim%")
   OR ("additionalDetail2" LIKE "%Tamil Nadu%")
   OR ("additionalDetail2" LIKE "%Telangana%")
   OR ("additionalDetail2" LIKE "%Tripura%")
   OR ("additionalDetail2" LIKE "%Uttar Pradesh%")
   OR ("additionalDetail2" LIKE "%Uttarakhand%")
   OR ("additionalDetail2" LIKE "%West Bengal%")
   OR ("additionalDetail2" LIKE "%Delhi%")
   OR ("additionalDetail2" LIKE "%Jammu and Kashmir%")
   OR ("additionalDetail2" LIKE "%Ladakh%")));""",
    },
    {
        "input": "How many scientific names are common to Japan and India?",
        "query": """
        SELECT COUNT(DISTINCT "scientificName")
FROM plants
WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN
 (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Japan%" AND
 (("additionalDetail2" LIKE "%India%")
   OR ("additionalDetail2" LIKE "%Andhra Pradesh%")
   OR ("additionalDetail2" LIKE "%Arunachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Assam%")
   OR ("additionalDetail2" LIKE "%Bihar%")
   OR ("additionalDetail2" LIKE "%Chhattisgarh%")
   OR ("additionalDetail2" LIKE "%Goa%")
   OR ("additionalDetail2" LIKE "%Gujarat%")
   OR ("additionalDetail2" LIKE "%Haryana%")
   OR ("additionalDetail2" LIKE "%Himachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Jharkhand%")
   OR ("additionalDetail2" LIKE "%Karnataka%")
   OR ("additionalDetail2" LIKE "%Kerala%")
   OR ("additionalDetail2" LIKE "%Madhya Pradesh%")
   OR ("additionalDetail2" LIKE "%Maharashtra%")
   OR ("additionalDetail2" LIKE "%Manipur%")
   OR ("additionalDetail2" LIKE "%Meghalaya%")
   OR ("additionalDetail2" LIKE "%Mizoram%")
   OR ("additionalDetail2" LIKE "%Nagaland%")
   OR ("additionalDetail2" LIKE "%Odisha%")
   OR ("additionalDetail2" LIKE "%Punjab%")
   OR ("additionalDetail2" LIKE "%Rajasthan%")
   OR ("additionalDetail2" LIKE "%Sikkim%")
   OR ("additionalDetail2" LIKE "%Tamil Nadu%")
   OR ("additionalDetail2" LIKE "%Telangana%")
   OR ("additionalDetail2" LIKE "%Tripura%")
   OR ("additionalDetail2" LIKE "%Uttar Pradesh%")
   OR ("additionalDetail2" LIKE "%Uttarakhand%")
   OR ("additionalDetail2" LIKE "%West Bengal%")
   OR ("additionalDetail2" LIKE "%Delhi%")
   OR ("additionalDetail2" LIKE "%Jammu and Kashmir%")
   OR ("additionalDetail2" LIKE "%Ladakh%")));
        """
    },
    {
        "input": "How many accepted names are common to Australia and India?",
        "query": """
        SELECT COUNT(DISTINCT "scientificName")
FROM plants
WHERE "recordType" = "AN" AND ("familyNumber", "genusNumber", "acceptedNameNumber") IN
 (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Australia%" AND
 (("additionalDetail2" LIKE "%India%")
   OR ("additionalDetail2" LIKE "%Andhra Pradesh%")
   OR ("additionalDetail2" LIKE "%Arunachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Assam%")
   OR ("additionalDetail2" LIKE "%Bihar%")
   OR ("additionalDetail2" LIKE "%Chhattisgarh%")
   OR ("additionalDetail2" LIKE "%Goa%")
   OR ("additionalDetail2" LIKE "%Gujarat%")
   OR ("additionalDetail2" LIKE "%Haryana%")
   OR ("additionalDetail2" LIKE "%Himachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Jharkhand%")
   OR ("additionalDetail2" LIKE "%Karnataka%")
   OR ("additionalDetail2" LIKE "%Kerala%")
   OR ("additionalDetail2" LIKE "%Madhya Pradesh%")
   OR ("additionalDetail2" LIKE "%Maharashtra%")
   OR ("additionalDetail2" LIKE "%Manipur%")
   OR ("additionalDetail2" LIKE "%Meghalaya%")
   OR ("additionalDetail2" LIKE "%Mizoram%")
   OR ("additionalDetail2" LIKE "%Nagaland%")
   OR ("additionalDetail2" LIKE "%Odisha%")
   OR ("additionalDetail2" LIKE "%Punjab%")
   OR ("additionalDetail2" LIKE "%Rajasthan%")
   OR ("additionalDetail2" LIKE "%Sikkim%")
   OR ("additionalDetail2" LIKE "%Tamil Nadu%")
   OR ("additionalDetail2" LIKE "%Telangana%")
   OR ("additionalDetail2" LIKE "%Tripura%")
   OR ("additionalDetail2" LIKE "%Uttar Pradesh%")
   OR ("additionalDetail2" LIKE "%Uttarakhand%")
   OR ("additionalDetail2" LIKE "%West Bengal%")
   OR ("additionalDetail2" LIKE "%Delhi%")
   OR ("additionalDetail2" LIKE "%Jammu and Kashmir%")
   OR ("additionalDetail2" LIKE "%Ladakh%")));
        """
    },
    {
        "input": "How many species are cultivated in India?",
        "query": """
        SELECT COUNT(DISTINCT "scientificName")
FROM plants
WHERE "recordType" IN ("AN", "SN") AND ("familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber") IN
 (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber", "synonymNumber" FROM plants WHERE "recordType" LIKE "%RE%" AND ("additionalDetail2" LIKE "%Cultivated%") AND
 (("additionalDetail2" LIKE "%India%")
   OR ("additionalDetail2" LIKE "%Andhra Pradesh%")
   OR ("additionalDetail2" LIKE "%Arunachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Assam%")
   OR ("additionalDetail2" LIKE "%Bihar%")
   OR ("additionalDetail2" LIKE "%Chhattisgarh%")
   OR ("additionalDetail2" LIKE "%Goa%")
   OR ("additionalDetail2" LIKE "%Gujarat%")
   OR ("additionalDetail2" LIKE "%Haryana%")
   OR ("additionalDetail2" LIKE "%Himachal Pradesh%")
   OR ("additionalDetail2" LIKE "%Jharkhand%")
   OR ("additionalDetail2" LIKE "%Karnataka%")
   OR ("additionalDetail2" LIKE "%Kerala%")
   OR ("additionalDetail2" LIKE "%Madhya Pradesh%")
   OR ("additionalDetail2" LIKE "%Maharashtra%")
   OR ("additionalDetail2" LIKE "%Manipur%")
   OR ("additionalDetail2" LIKE "%Meghalaya%")
   OR ("additionalDetail2" LIKE "%Mizoram%")
   OR ("additionalDetail2" LIKE "%Nagaland%")
   OR ("additionalDetail2" LIKE "%Odisha%")
   OR ("additionalDetail2" LIKE "%Punjab%")
   OR ("additionalDetail2" LIKE "%Rajasthan%")
   OR ("additionalDetail2" LIKE "%Sikkim%")
   OR ("additionalDetail2" LIKE "%Tamil Nadu%")
   OR ("additionalDetail2" LIKE "%Telangana%")
   OR ("additionalDetail2" LIKE "%Tripura%")
   OR ("additionalDetail2" LIKE "%Uttar Pradesh%")
   OR ("additionalDetail2" LIKE "%Uttarakhand%")
   OR ("additionalDetail2" LIKE "%West Bengal%")
   OR ("additionalDetail2" LIKE "%Delhi%")
   OR ("additionalDetail2" LIKE "%Jammu and Kashmir%")
   OR ("additionalDetail2" LIKE "%Ladakh%")));
        """,
    },
    {
        "input": "How many accepted names are only distributed in Karnataka?",
        "query": """SELECT COUNT(DISTINCT scientificName) as unique_pairs_count FROM plants WHERE "recordType" = "AN" AND
        ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants
        WHERE "recordType" = "DB" AND ("additionalDetail2" LIKE "%Karnataka%"
        AND NOT "additionalDetail2" LIKE "%Andhra Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Arunachal Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Assam%"
        AND NOT "additionalDetail2" LIKE "%Bihar%"
        AND NOT "additionalDetail2" LIKE "%Chhattisgarh%"
        AND NOT "additionalDetail2" LIKE "%Goa%"
        AND NOT "additionalDetail2" LIKE "%Gujarat%"
        AND NOT "additionalDetail2" LIKE "%Haryana%"
        AND NOT "additionalDetail2" LIKE "%Himachal Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Jharkhand%"
        AND NOT "additionalDetail2" LIKE "%Kerala%"
        AND NOT "additionalDetail2" LIKE "%Madhya Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Maharashtra%"
        AND NOT "additionalDetail2" LIKE "%Manipur%"
        AND NOT "additionalDetail2" LIKE "%Meghalaya%"
        AND NOT "additionalDetail2" LIKE "%Mizoram%"
        AND NOT "additionalDetail2" LIKE "%Nagaland%"
        AND NOT "additionalDetail2" LIKE "%Odisha%"
        AND NOT "additionalDetail2" LIKE "%Punjab%"
        AND NOT "additionalDetail2" LIKE "%Rajasthan%"
        AND NOT "additionalDetail2" LIKE "%Sikkim%"
        AND NOT "additionalDetail2" LIKE "%Tamil Nadu%"
        AND NOT "additionalDetail2" LIKE "%Telangana%"
        AND NOT "additionalDetail2" LIKE "%Tripura%"
        AND NOT "additionalDetail2" LIKE "%Uttar Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Uttarakhand%"
        AND NOT "additionalDetail2" LIKE "%West Bengal%"
        AND NOT "additionalDetail2" LIKE "%Delhi%"
        AND NOT "additionalDetail2" LIKE "%Jammu and Kashmir%"
        AND NOT "additionalDetail2" LIKE "%Ladakh%"));""",
    },
    {
        "input": "How many accepted names are distributed only in Assam?",
        "query": """SELECT COUNT(DISTINCT scientificName) FROM plants WHERE "recordType" = "AN" AND
        ("familyNumber", "genusNumber", "acceptedNameNumber") IN (SELECT DISTINCT "familyNumber", "genusNumber", "acceptedNameNumber" FROM plants
        WHERE "recordType" = "DB" AND "additionalDetail2" LIKE "%Assam%"
        AND NOT "additionalDetail2" LIKE "%Andhra Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Arunachal Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Bihar%"
        AND NOT "additionalDetail2" LIKE "%Chhattisgarh%"
        AND NOT "additionalDetail2" LIKE "%Goa%"
        AND NOT "additionalDetail2" LIKE "%Gujarat%"
        AND NOT "additionalDetail2" LIKE "%Haryana%"
        AND NOT "additionalDetail2" LIKE "%Himachal Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Jharkhand%"
        AND NOT "additionalDetail2" LIKE "%Kerala%"
        AND NOT "additionalDetail2" LIKE "%Madhya Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Maharashtra%"
        AND NOT "additionalDetail2" LIKE "%Manipur%"
        AND NOT "additionalDetail2" LIKE "%Meghalaya%"
        AND NOT "additionalDetail2" LIKE "%Mizoram%"
        AND NOT "additionalDetail2" LIKE "%Nagaland%"
        AND NOT "additionalDetail2" LIKE "%Odisha%"
        AND NOT "additionalDetail2" LIKE "%Punjab%"
        AND NOT "additionalDetail2" LIKE "%Rajasthan%"
        AND NOT "additionalDetail2" LIKE "%Sikkim%"
        AND NOT "additionalDetail2" LIKE "%Tamil Nadu%"
        AND NOT "additionalDetail2" LIKE "%Telangana%"
        AND NOT "additionalDetail2" LIKE "%Tripura%"
        AND NOT "additionalDetail2" LIKE "%Uttar Pradesh%"
        AND NOT "additionalDetail2" LIKE "%Uttarakhand%"
        AND NOT "additionalDetail2" LIKE "%West Bengal%"
        AND NOT "additionalDetail2" LIKE "%Delhi%"
        AND NOT "additionalDetail2" LIKE "%Jammu and Kashmir%"
        AND NOT "additionalDetail2" LIKE "%Ladakh%");"""
    },
    {"input": "List all the accepted names under the family 'Gnetaceae'.",
     "query": """
SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "familyNumber" IN (
        SELECT DISTINCT "familyNumber" FROM plants WHERE "familyName" = "Gnetaceae"
    )
) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
"""},
    {
        "input": "List all the accepted species that are introduced.",
        "query": """
SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "recordType" = 'RE'and "additionalDetail2" LIKE '%introduced%'
) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
""",
    },
    {
        "input": "List all the accepted names with type 'Cycad'",
        "query": """
SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "recordType" = 'HB' and "additionalDetail2" LIKE '%Cycad%'

) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
""",
    },
    {
        "input": "List all the accepted names under the genus 'Cycas' with more than two synonyms.",
        "query": """
SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "genusNumber" IN (
        SELECT DISTINCT "genusNumber" FROM plants WHERE "genusName" = 'Cycas'
    )
    AND "familyNumber" IN (
        SELECT DISTINCT "familyNumber" FROM plants WHERE "genusName" = 'Cycas'
    )
    AND "synonymNumber" > 2
) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
""",
    },
 {
        "input":'List all the accepted names published in Asian J. Conservation Biol.',
        "query": """
    SELECT DISTINCT "scientificName"
    FROM plants
    WHERE "recordType" = 'AN' AND "publication" LIKE '%Asian J. Conservation Biol%';

""",
    },
 {
        "input": 'List all the accepted names linked with endemic tag.',
        "query": """
SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "recordType" = 'DB'and "additionalDetail2" LIKE '%Endemic%'

) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
""",
    },
 {
        "input": 'List all the accepted names that have no synonyms.' ,
        "query": """
SELECT  DISTINCT a."scientificName" FROM plants a
group by a."familyNumber",a."genusNumber",a."acceptedNameNumber"
HAVING  SUM(a."synonymNumber") = 0 AND a."acceptedNameNumber" > 0;
""",
    },
 {
        "input": 'List all the accepted names authored by Roxb.',
        "query": """
SELECT "scientificName"
FROM plants
WHERE "recordType" = 'AN'AND "authorName" LIKE '%Roxb%';
""",
    },
 {
        "input": 'List all genera within each family',
        "query": """
SELECT "familyName", "genusName"
FROM plants
WHERE "recordType" = 'GE';
""",
    },
     {
        "input": 'Did Minq. discovered Cycas ryumphii?',
        "query": """SELECT
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM plants as a
            WHERE a."scientificName" = 'Cycas rumphii'
            AND a."authorName" = 'Miq.'
        ) THEN 'TRUE'
        ELSE 'FALSE'
    END AS ExistsCheck;
"""},
	{
        "input": "List the names of plants from the family 'Rosaceae' having more than 3 synonyms.",
        "query": """SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "genusNumber" IN (
        SELECT DISTINCT "genusNumber" FROM plants WHERE "genusName" = 'Berberis'
    )
    AND "familyNumber" IN (
        SELECT DISTINCT "familyNumber" FROM plants WHERE "genusName" = 'Berberis'
    )
    AND "synonymNumber" > 3
) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
"""
    },
    {
        "input": "List the names of plants from the family 'Cabombaceae' having more than 1 synonyms.",
        "query": """SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "familyNumber" IN (
        SELECT DISTINCT "familyNumber" FROM plants WHERE "familyName" = 'Cabombaceae'
    )
    AND "synonymNumber" > 1
) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
"""
    },
  {
        "input": "List the names of plants from the family 'Nymphaeaceae' having atleast one synonym.",
        "query": """SELECT DISTINCT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber"
    FROM plants
    WHERE "familyNumber" IN (
        SELECT DISTINCT "familyNumber" FROM plants WHERE "familyName" = 'Nymphaeaceae'
    )
    AND "synonymNumber" > 1
) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
"""
  },
  {
        "input": "List all genera within the family 'Polygalaceae'.",
        "query": """SELECT DISTINCT "genusName" FROM plants
    WHERE "familyName" = 'Polygalaceae'
    AND "recordType" = 'GE';
"""
  },
  {
      "input": "List the species that do not have any synonyms",
      "query": """
SELECT  DISTINCT a."Scientific_Name" FROM plants a
group by a."Family_Number",a."Genus_Number",a."Accepted_name_number"
HAVING  SUM(a."Synonym_Number") = 0 AND a."Accepted_name_number" > 0;
"""
  },
  {
      "input": "Find the family with the highest number of genera.",
      "query": """
SELECT "familyName", COUNT(DISTINCT "genusName") AS genera_count
FROM plants
WHERE "recordType" = 'GE'
GROUP BY "familyName"
ORDER BY genera_count DESC
LIMIT 1;"""
  },
  {
      "input":"Count the nnumber of species in Muraltia genus",
      "query": """
SELECT COUNT("scientificName")
FROM plants
WHERE "recordType" = 'AN'
AND "genusName" = 'Muraltia';"""
  },
  {
      "input":"Find the genus with the most species",
      "query": """
SELECT "genusName", COUNT(DISTINCT "scientificName") AS plant_count
FROM plants
WHERE "recordType" = 'AN'
GROUP BY "genusName"
ORDER BY plant_count DESC
LIMIT 1;"""
  },
  {
      "input":"List the accepted names for all plants in the genus 'Tribulus'",
      "query": """
SELECT "scientificName"
FROM plants
WHERE "recordType" = 'AN'
AND "genusName" = 'Tribulus';"""
  },
  {
      "input":"List the families that have more than 20 species.",
      "query": """
SELECT "familyName", COUNT(DISTINCT "scientificName") AS plant_count
FROM plants
WHERE "recordType" = 'AN'
GROUP BY "familyName"
HAVING plant_count > 20;"""
  },
  {
      "input":"Which family has the least number of species",
      "query": """
SELECT "familyName", COUNT(DISTINCT "scientificName") AS plant_count
FROM plants
WHERE "recordType" = 'AN'
GROUP BY "familyName"
ORDER BY plant_count ASC
LIMIT 1;"""
  },
  {
      "input":"List the genera with exactly 10 species.",
      "query": """
SELECT "genusName", COUNT(DISTINCT "scientificName") AS plant_count
FROM plants
WHERE "recordType" = 'AN'
GROUP BY "genusName"
HAVING plant_count = 10;"""
  },
  {
      "input": "Find the species in the genus 'Fissistigma' that have exactly 2 synonyms.",
      "query": """SELECT "scientificName" FROM plants
JOIN (
    SELECT "familyNumber", "genusNumber", "acceptedNameNumber", COUNT("synonymNumber") as s_count
    FROM plants
    WHERE "genusNumber" IN (
        SELECT DISTINCT "genusNumber" FROM plants WHERE "genusName" = 'Fissistigma'
    )
     AND "familyNumber" IN (
        SELECT DISTINCT "familyNumber" FROM plants WHERE "genusName" = 'Fissistigma'
    )
    AND "recordType" = 'SN'
    GROUP BY "acceptedNameNumber"
    HAVING s_count = 2
) b
ON plants."genusNumber" = b."genusNumber" AND plants."acceptedNameNumber" = b."acceptedNameNumber" AND plants."familyNumber" = b."familyNumber"
WHERE plants."recordType" = 'AN';
"""},
{
    "input": "List the genera that have at least one species with more than 10 synonyms.",
    "query": """SELECT DISTINCT p."genusName"
FROM plants p
JOIN (
    SELECT "acceptedNameNumber", "genusNumber", "familyNumber"
    FROM plants
    WHERE "synonymNumber" > 10
) s
ON p."acceptedNameNumber" = s."acceptedNameNumber" AND p."genusNumber" = s."genusNumber" AND p."familyNumber" = s."familyNumber"
WHERE p."recordType" = 'AN';
"""
},
]


		example_selector = SemanticSimilarityExampleSelector.from_examples(
				examples,
				OpenAIEmbeddings(),
				FAISS,
				k=5,
				input_keys=["input"],
				)

		prefix_prompt = """
		You are an agent designed to interact with a SQL database.
		Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
		You can order the results by a relevant column to return the most interesting examples in the database.
		Never query for all the columns from a specific table, only ask for the relevant columns given the question.
		You have access to tools for interacting with the database.
		Only use the given tools. Only use the information returned by the tools to construct your final answer.
		You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

		- Restrict your queries to the "plants" table.
		- Do not return more than {top_k} rows unless specified otherwise.
		- Add a limit of 25 at the end of SQL query.
		- If the SQLite query returns zero rows, return a message indicating the same.
		- Only refer to the data contained in the {table_info} table. Do not fabricate any data.
		- For filtering based on string comparison, always use the LIKE operator and enclose the string in `%`.
		- Queries on the `Additional_Details_2` column should use sub-queries involving `Family_Number`, `Genus_Number` and `Accepted_name_number`.

		Refer to the table description below for more details on the columns:
		1. **Record_Type_Code**: Contains text codes indicating the type of information in the row.
		- FA: Family Name, Genus Name, Scientific Name
		- TY: Type
		- GE: Genus Name
		- AN: Family Name (Accepted Name), Genus Name, Scientific Name, Author Name, Publication, Volume:Page, Year of Publication
		- HB: Habit
		- DB: Distribution/location of the plant
		- RE: Remarks
		- SN: Family Name (Synonym), Genus Name, Scientific Name, Author Name, Publication, Volume:Page, Year of Publication
		2. **Family_Name**: Contains the Family Name of the plant.
		3. **Genus_Name**: Contains the Genus Name of the plant.
		4. **Scientific_Name**: Contains the Scientific Name of the plant species.
		5. **Publication_Name**: Name of the journal or book where the plant discovery information is published. Use LIKE for queries.
		6. **Volume:_Page**: The volume and page number of the publication.
		7. **Year_of_Publication**: The year in which the plant information was published.
		8. **Author_Name**: May contain multiple authors separated by `&`. Use LIKE for queries.
		9. **Additional_Details**: Contains type, habit, distribution, and remarks. Use LIKE for queries.
		- Type: General location information.
		- Remarks: Location information about cultivation or native area.
		- Distribution: Locations where the plant is common. May contain multiple locations, use LIKE for queries.
		10. **Groups**: Contains either "Gymnosperms" or "Angiosperms".
		11. **Urban**: Contains either "YES" or "NO". Specifies whether the plant is urban.
		12. **Fruit**: Contains either "YES" or "NO". Specifies whether the plant is a fruit plant.
		13. **Medicinal**: Contains either "YES" or "NO". Specifies whether the plant is medicinal.
		14. **Genus_Number**: Contains the Genus Number of the plant.
		15. **Accepted_name_number**: Contains the Accepted Name Number of the plant.
		Remember "Only accepted name" means zero number of synonyms or no synonyms.
		Below are examples of questions and their corresponding SQL queries.
		Lastly, for queries related to accepted names, synonyms, genus, families try to use the examples given below and just replace the values such as fmaily name, genus name,etc  based on the input. Do not change anyother aspect of the query. 
		"""



		agent_prompt = PromptTemplate.from_template("User input: {input}\nSQL Query: {query}")
		agent_prompt_obj = FewShotPromptTemplate(
				example_selector=example_selector,
				example_prompt=agent_prompt,
				prefix=prefix_prompt,
				suffix="",
				input_variables=["input"],
		)

		full_prompt = ChatPromptTemplate.from_messages(
				[
						SystemMessagePromptTemplate(prompt=agent_prompt_obj),
						("human", "{input}"),
						MessagesPlaceholder("agent_scratchpad"),
				]
		)
		return full_prompt

def initalize_sql_agent(llm, db):

		dynamic_few_shot_prompt = get_dynamic_prompt_template()

		agent = create_sql_agent(llm, db=db, prompt=dynamic_few_shot_prompt, agent_type="openai-tools", verbose=True)

		return agent

def generate_response_agent(agent,user_question):
	response = agent.invoke({"input": user_question})
	return response