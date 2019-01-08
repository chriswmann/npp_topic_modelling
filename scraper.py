#!/usr/bin/env python3

from PIL import Image as Img
from PIL import ImageEnhance, ImageFilter
from scrapy.http import Request
from scrapy.spiders import Spider
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from wand.image import Image as WImage

import os
import pytesseract
import urllib.parse

log_file = './scraper.log'

base_path = './scraped_docs/'

pdf_output_path = os.path.join(base_path, 'pdf_output')
if not os.path.isdir(pdf_output_path):
    os.makedirs(pdf_output_path)

pdf_output_path_list = []
pdf_path = base_path
density = 500

class HprScraper(Spider):
    name = "www.ukhpr1000.co.uk_PDF_crawler"

    allowed_domains = ["www.ukhpr1000.co.uk"]
    start_urls = ["http://www.ukhpr1000.co.uk/documents-library/step-2/"]

    def parse(self, response):
        base_url = 'http://www.ukhpr1000.co.uk/documents-library/step-2/'
        for a in response.xpath('//a[contains(@href,".pdf")]/@href'):
            link = a.extract()
            link = urllib.parse.urljoin(base_url, link)
            yield Request(link, callback=self.save_pdf)

    def save_pdf(self, response):
        path = os.path.join(base_path, response.url.split('/')[-1])
        with open(path, 'wb') as f:
            f.write(response.body)

process = CrawlerProcess(get_project_settings())
process.crawl(HprScraper)
process.start()

print('Performing OCR on found PDFs...')
print('Image path is: {}'.format(pdf_output_path))
print('PDF path is: {}'.format(pdf_path))

job_length = len([f for f in os.listdir(pdf_path) if f.endswith('pdf')])
progress_pdf = 0
progress_img = 0
pdf_count = 0
for f in os.listdir(pdf_path):
    if f.endswith('pdf'):
        #Force newline so last progress update is not overwritten in the console
        print('')
        pdf = os.path.join(pdf_path, f)
        image_basename = os.path.basename(pdf).rstrip('.pdf')
        image_path = os.path.join(pdf_output_path, image_basename + os.sep)
        if not os.path.isdir(image_path):
            print('Output directory "{}" does not exist, creating it now...'.format(image_path))
            os.makedirs(image_path)
        print('Reading {}...'.format(pdf))
        with WImage(filename=pdf,resolution=density) as source:
            images = source.sequence
            pages=len(images)
            print('Writing {}...'.format(image_basename))
            for i in range(pages):
                path_to_file = os.path.join(image_path, image_basename + '_' + str(i)+'.jpg')
                WImage(images[i]).save(filename = path_to_file)
                pdf_output_path_list.append(path_to_file)
                progress_img = (i / pages) * 100
                print('**********Image Write Progress: {0:.0f}% **********'.format(progress_img), end='\r')
            #Force newline so last progress update is not overwritten in the console
            print('')
        pdf_count += 1
        progress_pdf = (pdf_count / job_length) * 100
        print('**********PDF Read progress: {0:.0f}% **********'.format(progress_pdf), end='\r')

for f in os.listdir(pdf_output_path):
    if f.endswith('jpg'):
        pdf_output_path_list.append(os.path.join(pdf_output_path, f))

print('Writing PDF text to disc...')
job_length = len(pdf_output_path_list)
progress_txt = 0
txt_count = 0
for pdf_image_path in pdf_output_path_list:
    fname = pdf_image_path.split('\\')[-1].rstrip('.jpg') + '.txt'
    try:
        im = Img.open(pdf_image_path)
        im = im.convert('1') # convert to black and white
        txt = pytesseract.image_to_string(im)
    except OSError as error:
        with open(log_file, 'a') as fout:
            fout.write(error + '\n')
    with open(fname, 'w') as fout:
        fout.write(txt)
    txt_count += 1
    progress_txt = (txt_count / job_length) * 100
    print('**********Text Write Progress: {0:.0f}% **********'.format(progress_txt), end='\r')
