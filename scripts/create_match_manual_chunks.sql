create or replace function public.match_manual_chunks(
    query_embedding vector(1024),
    match_threshold float,
    match_count int
)
returns table (
    chunk_id text,
    content text,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        vec_manuals.chunk_id,
        vec_manuals.content,
        1 - (vec_manuals.embedding <=> query_embedding) as similarity
    from
        vec_manuals
    where
        (vec_manuals.embedding <=> query_embedding) > match_threshold
    order by
        vec_manuals.embedding <=> query_embedding
    limit match_count;
end;
$$;