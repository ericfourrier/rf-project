#!/usr/bin/env python
# -*- coding: utf-8 -*-u

"""
Purpose : Get sppot price from aws instances via boto and aws we

http://info.awsstream.com : website to get some infos
"""

# describe_spot_datafeed_subscription()
# describe_spot_fleet_instances()
# describe_spot_fleet_request_history()
# describe_spot_fleet_requests()
# describe_spot_instance_requests()
# describe_spot_price_history()
import boto3
import logging
import json
import codecs
import os
import pandas as pd
import datetime

logger = logging.getLogger('aws')
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s -  %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from multiprocessing import Pool, cpu_count


# Json logging
# from pythonjsonlogger import jsonlogger
# logger = logging.getLogger()
# logHandler = logging.StreamHandler()
# formatter = jsonlogger.JsonFormatter()
# logHandler.setFormatter(formatter)
# logger.addHandler(logHandler)

client = boto3.client('ec2')  # youc an put a region in a client


def get_regions_ec2():
    """ Returns list of regions available on ec2 """
    return [d['RegionName'] for d in boto3.client('ec2').describe_regions()['Regions']]
#price_hist_micro = client.describe_spot_price_history(InstanceTypes=['t1.micro'])

colnames_df = ['AvailabilityZone', 'ProductDescription', 'SpotPrice',
               'InstanceType', 'Timestamp']


def write_to_json_line(data, file_path):
    """ Save one object to json (one object per line) """
    with codecs.open(file_path, 'a', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write(os.linesep)


def help_wjson(l, file_path):
    """ write to json a list of same data"""
    for e in l:
        e['Timestamp'] = str(e['Timestamp'])
        write_to_json_line(e, file_path)


def get_brute_force_sp(client=client):
    req_first = client.describe_spot_price_history()
    next_token = req_first['NextToken']
    # store first request in list_results
    list_results = req_first['SpotPriceHistory']
    count = 1000
    while (next_token is not None and isinstance(next_token, str)):
        req = client.describe_spot_price_history(NextToken=next_token)
        next_token = req['NextToken']
        list_results += req['SpotPriceHistory']
        count += len(req['SpotPriceHistory'])
        logger.info("Nb of spot prices downloaded : {}".format(count))
        logger.info("Next token : {}".format(next_token))
    return list_results


def get_sp_all_json(client=client, file_path='spot_price_aws.json'):
    req_first = client.describe_spot_price_history()
    next_token = req_first['NextToken']
    # store first request in list_results
    help_wjson(req_first['SpotPriceHistory'], file_path)
    count = 1000
    while (next_token is not None and isinstance(next_token, str)):
        req = client.describe_spot_price_history(NextToken=next_token)
        next_token = req['NextToken']
        help_wjson(req['SpotPriceHistory'], file_path)
        count += len(req['SpotPriceHistory'])
        logger.info("Nb of spot prices downloaded : {}".format(count))
        logger.info("Next token : {}".format(next_token))


def get_sp_all_csv_old(client=client, file_path='spot_price_aws_2.csv',
                       end_time=datetime.datetime.now(), *args, **kwargs):

    visited_token = set()
    req_first = client.describe_spot_price_history(
        EndTime=end_time, *args, **kwargs)
    next_token = req_first['NextToken']
    visited_token.add(next_token)
    pd.DataFrame(req_first['SpotPriceHistory']).to_csv(file_path, index=False)
    count = 1000
    while (next_token is not None and isinstance(next_token, str)):
        req = client.describe_spot_price_history(EndTime=end_time,
                                                 NextToken=next_token, *args, **kwargs)
        next_token = req['NextToken']
        if next_token in visited_token:
            break
        else:
            visited_token.add(next_token)
            df = pd.DataFrame(req['SpotPriceHistory'])
            df.to_csv(file_path, index=False, header=False, mode='a')
            count += len(req['SpotPriceHistory'])
            logger.info({'nb_sp_downloaded': count, 'next_token': next_token})

#
# paginator = ec2.get_paginator('describe_spot_price_history')
# for page in paginator.paginate(StartTime=start_time, EndTime=end_time):
#     price_history += page['SpotPriceHistory']


def get_sp_all_csv(region_name, file_path='test.csv', *args, **kwargs):
    count = 0
    # write for colname
    pd.DataFrame(columns=colnames_df).to_csv(file_path, index=False)
    c = boto3.client('ec2', region_name=region_name)
    paginator = c.get_paginator('describe_spot_price_history')
    for p in paginator.paginate():
        pd.DataFrame(p['SpotPriceHistory']).to_csv(file_path, index=False, header=False, mode='a')
        #count += len(p['SpotPriceHistory'])
        # logger.info('nb_sp_downloaded:{}'.format(count))


def get_sp_all_csv_p(file_path='Data_aws/spot_price_aws_total', **kws):
    """ Multi core version of get_sp_all_csv_old"""
    # write colnames to file
    #pd.DataFrame(columns=colnames_df).to_csv(file_path, index=False)
    pool = Pool(cpu_count())
    region_names = get_regions_ec2()
    for r in region_names:
        fp = file_path + '_' + r + '.csv'
        pool.apply_async(get_sp_all_csv, args=(r, fp), **kws)
    pool.close()
    pool.join()
